"""Select a formula retrieval weight using validation data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf

from analytical_fm.configuration import DEFAULT_SETTINGS
from analytical_fm.data.datamodules import TTTMultiModalDataModule
from analytical_fm.data.datasets import build_dataset_multimodal
from analytical_fm.modeling.multitask_retrieval import (
    MultimodalCandidateRetriever,
    encode_formula_compositions,
)
from analytical_fm.modeling.wrapper import HFWrapper
from analytical_fm.utils import seed_everything


def _plain_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    return dict(value)


def _group_bootstrap_se(
    per_query_values: np.ndarray,
    groups: Sequence[object],
    *,
    n_bootstrap: int,
    seed: int,
) -> float:
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    if len(unique_groups) <= 1 or n_bootstrap <= 1:
        return 0.0
    group_rows = {
        group: np.flatnonzero(group_array == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_rows = np.concatenate([group_rows[group] for group in sampled_groups])
        means[bootstrap_index] = float(per_query_values[sampled_rows].mean())
    return float(means.std(ddof=1))


def _selected_tanimoto(
    query_fingerprints: torch.Tensor,
    candidate_fingerprints: torch.Tensor,
    selected_indices: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    selected = candidate_fingerprints[selected_indices]
    query = query_fingerprints[:, None, :]
    intersection = (query * selected).sum(dim=-1)
    union = query.sum(dim=-1) + selected.sum(dim=-1) - intersection
    return intersection / union.clamp_min(eps)


def _load_validation_groups(data_path: str, expected_length: int) -> np.ndarray:
    parquet_path = Path(data_path) / "validation.parquet"
    try:
        groups = pd.read_parquet(parquet_path, columns=["mces_cluster"])[
            "mces_cluster"
        ].to_numpy()
    except (FileNotFoundError, KeyError):
        groups = np.arange(expected_length)
    if len(groups) != expected_length:
        raise ValueError(
            f"Expected {expected_length} validation groups, found {len(groups)}."
        )
    return groups


def _float_predictions(predictions: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "fused_fingerprint_logits": predictions[
            "fused_fingerprint_logits"
        ].float(),
        "task_logits": {
            name: value.float()
            for name, value in predictions["task_logits"].items()
        },
        "modality_weights": predictions["modality_weights"].float(),
        "modality_order": predictions["modality_order"],
    }


@hydra.main(
    version_base=None,
    config_path=DEFAULT_SETTINGS.configs_path,
    config_name="config_train",
)
def main(config: DictConfig) -> None:
    proxy_config = _plain_mapping(config.get("formula_proxy"))
    query_split = str(proxy_config.get("query_split", "validation"))
    if query_split != "validation":
        raise ValueError("Formula retrieval weight selection is restricted to validation.")

    weights = sorted(
        {float(weight) for weight in proxy_config.get("weights", [0, 0.25, 0.5, 1, 2])}
    )
    if not weights or weights[0] != 0.0 or any(weight < 0 for weight in weights):
        raise ValueError("Formula proxy weights must be non-negative and include 0.")
    top_k = int(proxy_config.get("top_k", 512))
    minimum_improvement = float(proxy_config.get("minimum_improvement", 0.01))
    minimum_unique_ratio = float(proxy_config.get("minimum_unique_ratio", 0.9))
    n_bootstrap = int(proxy_config.get("n_bootstrap", 2000))

    seed_everything(seed=int(config["seed"]))
    output_dir = Path(config["working_dir"]) / config["job_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_data_config = OmegaConf.to_container(
        config["data"].copy(), resolve=True
    )
    model_config: Dict[str, Any] = dict(
        OmegaConf.to_container(config["model"].copy(), resolve=True) or {}
    )
    data_config, full_dataset = build_dataset_multimodal(
        initial_data_config,
        data_path=config["data_path"],
        cv_split=config["cv_split"],
        splitting=config["splitting"],
        augment_config=config["augment"],
        num_cpu=config["num_cpu"],
        mixture_config=config["mixture"],
    )
    dataset = DatasetDict(
        {
            "train": full_dataset["train"],
            "validation": full_dataset["validation"],
            "test": full_dataset[query_split],
        }
    )

    preprocessor_path = Path(config["preprocessor_path"])
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
    data_config, preprocessors = pd.read_pickle(preprocessor_path)
    if "MSMS" in preprocessors and preprocessors["MSMS"].max_sequence_length > 924:
        preprocessors["MSMS"].max_sequence_length = (
            int(model_config["max_position_embeddings"]) - 100
        )

    target_modality = "Smiles"
    model = HFWrapper(
        data_config=data_config,
        target_tokenizer=preprocessors[target_modality],
        num_steps=1,
        modality_dropout=config["modality_dropout"],
        **model_config,
    )
    checkpoint_path = Path(model_config["model_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    requested_device = str(proxy_config.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = requested_device
    model.eval()
    model.to(device)

    data_module = TTTMultiModalDataModule(
        model=model,
        dataset=dataset,
        preprocessors=preprocessors,
        data_config=data_config,
        model_type=model_config["model_type"],
        batch_size=int(model_config["batch_size"]),
        num_workers=int(config["num_cpu"]),
        extra_columns=[config["predict_class"]],
        device=device,
        reduced_val=False,
        similarity_criterion="multitask_retrieval",
    )
    autocast_enabled = device == "cuda" and bool(proxy_config.get("bf16", True))
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        predictions = data_module.get_multitask_predictions(
            data_module.datamodule.predict_dataloader()
        )
    predictions = _float_predictions(predictions)
    candidate_fingerprints, candidate_tasks = data_module.get_candidate_retrieval_bank(
        "train"
    )
    query_fingerprints, _ = data_module.get_candidate_retrieval_bank("test")
    candidate_fingerprints = candidate_fingerprints.float()
    candidate_tasks = {name: value.float() for name, value in candidate_tasks.items()}
    query_fingerprints = query_fingerprints.float()

    formula_modality = str(proxy_config.get("formula_modality", "Formula"))
    candidate_formulas = list(dataset["train"][formula_modality])
    query_formulas = list(dataset["test"][formula_modality])
    formula_counts, formula_elements = encode_formula_compositions(
        candidate_formulas + query_formulas
    )
    n_candidates = len(candidate_formulas)
    candidate_formula_counts = formula_counts[:n_candidates]
    query_formula_counts = formula_counts[n_candidates:]

    retrieval_config = _plain_mapping(config["activeft"].get("retrieval"))
    fingerprint_weight = float(retrieval_config.get("fingerprint_weight", 1.0))
    task_weights = dict(
        retrieval_config.get(
            "task_weights", {"MSMS": 1.0, "NMR": 1.0, "IR": 1.0}
        )
    )
    formula_similarity = str(
        proxy_config.get(
            "formula_similarity",
            retrieval_config.get("formula_similarity", "cosine"),
        )
    )
    groups = _load_validation_groups(config["data_path"], len(query_fingerprints))

    results: Dict[str, Dict[str, Any]] = {}
    per_query_means: Dict[float, np.ndarray] = {}
    for weight in weights:
        retriever = MultimodalCandidateRetriever(
            fingerprint_weight=fingerprint_weight,
            task_weights=task_weights,
            formula_weight=weight,
            formula_similarity=formula_similarity,
        )
        ranking = retriever.rank(
            predictions,
            candidate_fingerprints,
            candidate_tasks,
            top_k=min(top_k, len(candidate_fingerprints)),
            query_formula_counts=query_formula_counts,
            candidate_formula_counts=candidate_formula_counts,
        )
        if not bool(ranking.valid_mask.all()):
            raise RuntimeError(f"Non-finite retrieval score for formula weight {weight}.")
        selected_indices = ranking.indices.cpu()
        tanimoto = _selected_tanimoto(
            query_fingerprints, candidate_fingerprints, selected_indices
        )
        query_mean = tanimoto.mean(dim=1).numpy()
        query_best = tanimoto.max(dim=1).values.numpy()
        per_query_means[weight] = query_mean
        results[str(weight)] = {
            "mean_selected_tanimoto": float(query_mean.mean()),
            "mean_best_tanimoto": float(query_best.mean()),
            "fraction_best_ge_0_5": float((query_best >= 0.5).mean()),
            "selected_unique": int(torch.unique(selected_indices).numel()),
            "selection_exposures": int(selected_indices.numel()),
            "per_query_mean_selected_tanimoto": query_mean.tolist(),
        }

    baseline = results["0.0"]
    eligible_weights = [
        weight
        for weight in weights
        if weight > 0
        and results[str(weight)]["mean_selected_tanimoto"]
        >= baseline["mean_selected_tanimoto"] + minimum_improvement
        and results[str(weight)]["selected_unique"]
        >= baseline["selected_unique"] * minimum_unique_ratio
    ]
    selected_weight = 0.0
    selection_reason = "no non-zero weight passed the validation proxy gates"
    if eligible_weights:
        best_weight = max(
            eligible_weights,
            key=lambda weight: results[str(weight)]["mean_selected_tanimoto"],
        )
        best_se = _group_bootstrap_se(
            per_query_means[best_weight],
            groups,
            n_bootstrap=n_bootstrap,
            seed=int(config["seed"]),
        )
        one_se_floor = (
            results[str(best_weight)]["mean_selected_tanimoto"] - best_se
        )
        selected_weight = min(
            weight
            for weight in eligible_weights
            if results[str(weight)]["mean_selected_tanimoto"] >= one_se_floor
        )
        selection_reason = (
            "smallest eligible weight within one grouped-bootstrap SE of the best"
        )
    else:
        best_weight = 0.0
        best_se = _group_bootstrap_se(
            per_query_means[0.0],
            groups,
            n_bootstrap=n_bootstrap,
            seed=int(config["seed"]),
        )

    summary = {
        "query_split": query_split,
        "checkpoint": str(checkpoint_path),
        "n_train": len(candidate_fingerprints),
        "n_queries": len(query_fingerprints),
        "top_k": min(top_k, len(candidate_fingerprints)),
        "formula_similarity": formula_similarity,
        "formula_elements": formula_elements,
        "weights": weights,
        "minimum_improvement": minimum_improvement,
        "minimum_unique_ratio": minimum_unique_ratio,
        "selected_formula_weight": selected_weight,
        "best_proxy_weight": best_weight,
        "best_proxy_group_bootstrap_se": best_se,
        "selection_reason": selection_reason,
        "results": results,
    }
    summary_path = output_dir / "formula_validation_proxy.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "predictions": predictions,
            "candidate_fingerprints": candidate_fingerprints,
            "candidate_tasks": candidate_tasks,
            "query_fingerprints": query_fingerprints,
            "candidate_formula_counts": candidate_formula_counts,
            "query_formula_counts": query_formula_counts,
        },
        output_dir / "formula_validation_proxy_cache.pt",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
