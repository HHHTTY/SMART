import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as functional
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    GenerationConfig,
    PreTrainedModel,
    T5ForConditionalGeneration,
)
from transformers.generation.logits_process import LogitsProcessor
from transformers.modeling_outputs import Seq2SeqModelOutput

from ..generation.logit_processors import GuidedFormulaProcessor
from ..utils import calc_sampling_metrics
from .custom_bart_modeling import CustomBartConfig, CustomBartForConditionalGeneration
from .custom_modeling import AlignConfig, CustomConfig, CustomModel
from .exposure_ttt import ExposureTTTBlock
from .tahcd import TAHCDSharedBlock
from .spectral_repair import SpectralRepairBlock
from .generation_fusion import GenerationFusion
from .multilabel_metrics import MultilabelValidationMetrics
from .multitask_retrieval import TaskSpecializedRetrievalHeads
from .utils import CustomLMOutput, DummyLayer, MultimodalEmbedding, SincCosPositionalEncoding

OPTIMISER_REGISTRY = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def _sample_grouped_modality_subset(
    config: Mapping[str, Any],
    available_modalities: set[str],
) -> set[str]:
    """Sample modalities to drop for source-side subset multitask training.

    Formula is intentionally omitted from ``groups`` by the E6 configuration and
    is therefore always retained.  HNMR and CNMR can be treated as one atomic NMR
    group so that the requested single/pair/full spectroscopy mixture is sampled
    at the group level rather than at the raw encoder-input level.
    """
    policy = str(config.get("policy", ""))
    if policy != "spectroscopy_subset":
        raise ValueError(f"Unknown grouped modality dropout policy: {policy!r}")

    raw_groups = config.get("groups")
    if not isinstance(raw_groups, (Mapping, DictConfig)) or not raw_groups:
        raise ValueError("spectroscopy_subset requires a non-empty 'groups' mapping")
    groups = {
        str(name): [str(modality) for modality in modalities]
        for name, modalities in raw_groups.items()
    }
    controlled_modalities = {
        modality
        for modalities in groups.values()
        for modality in modalities
        if modality in available_modalities
    }
    if not controlled_modalities:
        raise ValueError("No configured spectroscopy modalities exist in this batch")

    single_probability = float(config.get("single_probability", 0.4))
    pair_probability = float(config.get("pair_probability", 0.3))
    full_probability = float(config.get("full_probability", 0.3))
    probabilities = np.asarray(
        [single_probability, pair_probability, full_probability], dtype=np.float64
    )
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError(
            "single_probability + pair_probability + full_probability must equal 1"
        )

    group_names = list(groups)
    bucket = int(np.random.choice(3, p=probabilities))
    keep_count = min((1, 2, len(group_names))[bucket], len(group_names))
    kept_group_names = set(
        str(name)
        for name in np.random.choice(group_names, keep_count, replace=False).tolist()
    )
    kept_modalities = {
        modality
        for name in kept_group_names
        for modality in groups[name]
        if modality in available_modalities
    }
    return controlled_modalities.difference(kept_modalities)


def _to_batch_first(value: Any) -> Any:
    """Transpose collator outputs from sequence-first to batch-first."""
    if isinstance(value, torch.Tensor):
        return value.transpose(1, 0) if value.ndim >= 2 else value
    if isinstance(value, dict):
        return {key: _to_batch_first(item) for key, item in value.items()}
    return value


def load_bart_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[BartForConditionalGeneration, MultimodalEmbedding]:
    """Loads a huggingface bart model.
    Args:
        model_name: Modelname e.g. facebook/bart-large
        target_tokenizer: target tokenizer for the target modality
        target_modality: Key of the target modality
        target_embedding_layer: embedding layer for the target modality
        kwargs: Additional model parameters
    Returns:
        BartForConditionalGeneration: Loaded model
    """

    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        bos_token_id=target_tokenizer.bos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        decoder_start_token_id=target_tokenizer.bos_token_id,
        forced_eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    bart_model = BartForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    bart_model.model.shared = multimodal_embedding_layer
    bart_model.model.encoder.embed_tokens = multimodal_embedding_layer
    bart_model.model.decoder.embed_tokens = (
        multimodal_embedding_layer.embedding_layer_dict[target_modality]
    )

    # Replace Layer Norm
    if multimodal_norm:
        dummy_layer = DummyLayer()
        bart_model.model.encoder.layernorm_embedding = dummy_layer

    # Replace learned pos embedding
    pos_embeds = SincCosPositionalEncoding(model_config.d_model)
    bart_model.model.encoder.embed_positions = pos_embeds
    bart_model.model.decoder.embed_positions = pos_embeds

    return bart_model, multimodal_embedding_layer

def load_custom_bart_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[CustomBartForConditionalGeneration, MultimodalEmbedding]:
    """Loads a huggingface bart model.
    Args:
        model_name: Modelname e.g. facebook/bart-large
        target_tokenizer: target tokenizer for the target modality
        target_modality: Key of the target modality
        target_embedding_layer: embedding layer for the target modality
        kwargs: Additional model parameters
    Returns:
        BartForConditionalGeneration: Loaded model
    """

    model_config = CustomBartConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        bos_token_id=target_tokenizer.bos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        decoder_start_token_id=target_tokenizer.bos_token_id,
        forced_eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    custom_bart_model = CustomBartForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    custom_bart_model.model.shared = multimodal_embedding_layer
    custom_bart_model.model.encoder.embed_tokens = multimodal_embedding_layer
    custom_bart_model.model.decoder.embed_tokens = (
        multimodal_embedding_layer.embedding_layer_dict[target_modality]
    )

    # Replace Layer Norm
    if multimodal_norm:
        dummy_layer = DummyLayer()
        custom_bart_model.model.encoder.layernorm_embedding = dummy_layer
        custom_bart_model.model.decoder.layernorm_embedding = dummy_layer
        #custom_bart_model.model.decoder.layernorm_embedding = dummy_layer

    # Replace learned pos embedding
    pos_embeds = SincCosPositionalEncoding(model_config.d_model)
    custom_bart_model.model.encoder.embed_positions = pos_embeds
    custom_bart_model.model.decoder.embed_positions = pos_embeds

    return custom_bart_model, multimodal_embedding_layer


def load_custom_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[CustomModel, MultimodalEmbedding]:
        
    config_kwargs = dict(
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        bos_token_id=target_tokenizer.bos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        decoder_start_token_id=target_tokenizer.bos_token_id,
        forced_eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )
    try:
        model_config = CustomConfig.from_pretrained(model_name, **config_kwargs)
    except (OSError, TimeoutError, ValueError):
        # The custom model is initialized from scratch; an online BART config
        # is only a legacy source of defaults and is not a weight dependency.
        model_config = CustomConfig(**config_kwargs)

    if model_config.align_config and not isinstance(model_config.align_config, AlignConfig):
        model_config.align_config = AlignConfig(**model_config.align_config)

    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm, do_positional_encodings=True, positional_encodings_type=model_config.positional_encoding_type, max_seq_len=model_config.max_position_embeddings
    )

    custom_model = CustomModel(target_modality, target_tokenizer, model_config, multimodal_embedding_layer)

    return custom_model, multimodal_embedding_layer
    

def load_t5_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[T5ForConditionalGeneration, MultimodalEmbedding]:

    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    t5_model = T5ForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    t5_model.shared = multimodal_embedding_layer
    t5_model.encoder.set_input_embeddings(multimodal_embedding_layer)

    if multimodal_norm:
        target_embedding = nn.Sequential(
            *[
                multimodal_embedding_layer.embedding_layer_dict[target_modality],
                multimodal_embedding_layer.embedding_norm_dict[target_modality],
            ]
        )
        t5_model.decoder.set_input_embeddings(target_embedding)
    else:
        target_embedding = multimodal_embedding_layer.embedding_layer_dict[
            target_modality
        ]
        t5_model.decoder.set_input_embeddings(target_embedding)

    return t5_model, multimodal_embedding_layer


MODEL_REGISTRY: Dict[str, Callable[..., Tuple[PreTrainedModel, MultimodalEmbedding]]] = {
    T5ForConditionalGeneration.__name__: load_t5_model,
    BartForConditionalGeneration.__name__: load_bart_model,
    CustomBartForConditionalGeneration.__name__: load_custom_bart_model,
    CustomModel.__name__: load_custom_model
}


class HFWrapper(pl.LightningModule):
    """Wrapper for Hugging Face models."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        model_type: str,
        model_name: str,
        target_tokenizer: Union[AutoTokenizer, str],
        optimiser: str = "adam",
        num_steps: int = 1000,
        lr: float = 0.001,
        weight_decay: float = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        multimodal_norm: bool = True,
        modality_dropout: Optional[List[str]] = None,
        excluded_input_modalities: Optional[List[str]] = None,
        generation_fusion_config: Optional[Dict[str, Any]] = None,
        multitask_retrieval_config: Optional[Dict[str, Any]] = None,
        exposure_ttt_config: Optional[Dict[str, Any]] = None,
        tahcd_config: Optional[Dict[str, Any]] = None,
        spectral_repair_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Init
        Args:
            data_config: Data configuration to set up modalities
            model_type: E.g. T5ForConditionGeneration
            model_name: model config to use e.g. google-t5/t5-small
            target_tokenizer: Either string or AutoTokenizer for the target modality. If string load via HF
            optimiser: Which optimiser to use. adam, adamw
            num_steps: Number of training steps
            lr: Learning rate
            weight_decay: weight decay
            adam_beta1: adam beta 1
            adam_beta2: adam beta 2
            multimodal_norm: Wether to apply layer norm to embedding or not
            kwargs: Additional Model parameters
        """
        super().__init__()

        # Wrapper Arguments
        if isinstance(target_tokenizer, str):
            self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
        else:
            self.target_tokenizer = target_tokenizer

        self.model_type = model_type
        self.model_name = model_name
        self.data_config = data_config
        self.multimodal_norm = multimodal_norm
        self.modality_dropout = modality_dropout
        self.excluded_input_modalities = frozenset(
            str(modality) for modality in (excluded_input_modalities or [])
        )
        unknown_exclusions = self.excluded_input_modalities.difference(data_config)
        if unknown_exclusions:
            raise ValueError(
                "Unknown excluded input modalities: "
                f"{sorted(unknown_exclusions)}"
        )
        self.generation_fusion_config = generation_fusion_config
        self.qmf_config: Dict[str, Any] = {}
        self.multitask_retrieval_config = multitask_retrieval_config
        self.exposure_ttt_config: Dict[str, Any] = dict(exposure_ttt_config or {})
        self.tahcd_config: Dict[str, Any] = dict(tahcd_config or {})
        self.spectral_repair_config: Dict[str, Any] = dict(spectral_repair_config or {})
        self.exposure_ttt: Optional[ExposureTTTBlock] = None
        self.tahcd: Optional[TAHCDSharedBlock] = None
        self.spectral_repair: Optional[SpectralRepairBlock] = None
        self._exposure_ttt_meta_batch_fraction = float(
            self.exposure_ttt_config.get("meta_batch_fraction", 0.3)
        )
        self._exposure_ttt_inference_enabled = bool(
            self.exposure_ttt_config.get("inference_ttt_enabled", False)
        )
        self._exposure_ttt_validation_compare = bool(
            self.exposure_ttt_config.get("validation_compare", True)
        )
        self._exposure_ttt_validation_controls = bool(
            self.exposure_ttt_config.get("validation_controls", True)
        )
        self._exposure_ttt_freeze_backbone = bool(
            self.exposure_ttt_config.get("freeze_backbone", False)
        )
        self._exposure_ttt_unfreeze_encoder_layers = int(
            self.exposure_ttt_config.get("unfreeze_encoder_layers", 0)
        )
        if self._exposure_ttt_unfreeze_encoder_layers < 0:
            raise ValueError("exposure_ttt_config.unfreeze_encoder_layers must be non-negative")
        if not 0.0 <= self._exposure_ttt_meta_batch_fraction <= 1.0:
            raise ValueError("exposure_ttt_config.meta_batch_fraction must be in [0, 1]")
        self.guided_generation = kwargs['guided_generation'] if 'guided_generation' in kwargs else False
        self.validation_n_beams = int(kwargs.get("validation_n_beams", 1))
        self.validation_guided_generation = bool(
            kwargs.get("validation_guided_generation", False)
        )
        self.lr_scheduler = kwargs['lr_scheduler'] if 'lr_scheduler' in kwargs else False
        self.gamma = kwargs['lr_gamma'] if 'lr_gamma' in kwargs else 0.95

        self.set_sel_indices: set = set()

        # Extract Target modality
        self.target_modality = ""
        for modality, modality_config in self.data_config.items():
            if modality_config["target"]:
                self.target_modality = modality

        # PL arguments
        self.optimiser = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.num_steps = num_steps

        self.train_step_outputs: List[Dict[str, Any]] = list()
        self.validation_step_outputs: List[Dict[str, Any]] = list()
        self.test_step_outputs: List[Dict[str, Any]] = list()

        self.hf_model, self.multimodal_embedding = MODEL_REGISTRY[self.model_type](
            self.model_name,
            self.target_tokenizer,
            self.target_modality,
            self.data_config,
            self.multimodal_norm,
            **kwargs,
        )

        self.multitask_retrieval_heads = None
        self.generation_fusion = None
        self.ms_validation_metrics = None
        self._ms_validation_sample_count = 0
        if self.generation_fusion_config is not None:
            fusion_config = dict(self.generation_fusion_config)
            self.qmf_config = dict(fusion_config.pop("qmf", {}) or {})
            self.generation_fusion = GenerationFusion(
                d_model=int(fusion_config.get("d_model", kwargs.get("d_model", 512))),
                variant=str(fusion_config.get("variant", "nmr_anchor_residual")),
                modality_map=fusion_config.get("modality_map"),
                dropout=float(fusion_config.get("dropout", 0.0)),
                read_n_heads=int(fusion_config.get("read_n_heads", 4)),
                quality_temperature=float(
                    fusion_config.get("quality_temperature", 1.0)
                ),
                selective_adapter_rank=int(
                    fusion_config.get("selective_adapter_rank", 16)
                ),
                selective_router_temperature=float(
                    fusion_config.get("selective_router_temperature", 0.001)
                ),
                selective_use_gumbel=bool(
                    fusion_config.get("selective_use_gumbel", True)
                ),
                selective_router_mode=str(
                    fusion_config.get("selective_router_mode", "global")
                ),
                selective_router_sample_scale=float(
                    fusion_config.get("selective_router_sample_scale", 1.0)
                ),
                selective_router_no_update_threshold=float(
                    fusion_config.get("selective_router_no_update_threshold", 0.85)
                ),
                selective_router_no_update_temperature=float(
                    fusion_config.get("selective_router_no_update_temperature", 0.05)
                ),
            )
            if self.qmf_config.get("enabled", False) and (
                self.generation_fusion.variant != "qmf_quality_router"
            ):
                raise ValueError(
                    "QMF training requires generation fusion variant "
                    "'qmf_quality_router'."
                )
        if self.multitask_retrieval_config is not None:
            retrieval_config = self.multitask_retrieval_config
            task_dimensions = dict(retrieval_config["task_dimensions"])
            encoder_dim = int(retrieval_config.get("encoder_dim", kwargs.get("d_model", 512)))
            self.multitask_retrieval_heads = TaskSpecializedRetrievalHeads(
                encoder_dim=encoder_dim,
                hidden_dim=int(retrieval_config["hidden_dim"]),
                fingerprint_dim=int(retrieval_config["fingerprint_dim"]),
                task_dimensions=task_dimensions,
                use_modality_attention=bool(
                    retrieval_config.get("use_modality_attention", False)
                ),
                modality_attention_heads=int(
                    retrieval_config.get("modality_attention_heads", 4)
                ),
                modality_attention_dropout=float(
                    retrieval_config.get("modality_attention_dropout", 0.0)
                ),
                use_formula_conditioning=bool(
                    retrieval_config.get("use_formula_conditioning", False)
                ),
            )
            if "MSMS" in task_dimensions:
                self.ms_validation_metrics = MultilabelValidationMetrics(
                    num_labels=int(task_dimensions["MSMS"])
                )

        # Use custom barebones generation config to avoid artifacts from logits_processors
        self.generation_config = GenerationConfig(
            bos_token_id=self.target_tokenizer.bos_token_id,
            decoder_start_token_id=self.target_tokenizer.bos_token_id,
            eos_token_id=self.target_tokenizer.eos_token_id,
            forced_eos_token_id=self.target_tokenizer.eos_token_id,
            max_length=128,  # Make variable at some point
            pad_token_id=self.target_tokenizer.pad_token_id,
        )

        self.n_beams = kwargs["n_beams"] if "n_beams" in kwargs else 10
        self._init_params()
        self._attach_exposure_ttt()
        self._attach_tahcd()
        self._attach_spectral_repair()

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for name, params in self.named_parameters():
            # Eq. (5) starts each Phi_m at the identity. Preserve the zero
            # residual/output and neutral semaphore initialization.
            if (
                name.endswith("selective_adapter_up")
                or name.endswith("selective_router_logits")
                or name.endswith("modality_adaptors")
                or name.endswith("shift_semaphore")
                or name.startswith("tahcd.")
                or name.startswith("spectral_repair.")
            ):
                continue
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

    def _attach_tahcd(self) -> None:
        """Attach the shared-encoder TAHCD transfer block when requested."""
        config = self.tahcd_config
        if not bool(config.get("enabled", False)):
            return
        d_model = int(getattr(self.hf_model.config, "d_model", 0))
        modality_map = dict(
            config.get(
                "modality_map",
                {name: name for name in ("Formula", "HNMR", "CNMR", "MSMS", "IR")},
            )
        )
        self.tahcd = TAHCDSharedBlock(
            d_model=d_model,
            modalities=tuple(str(name) for name in config.get("modalities", ("HNMR", "MSMS", "IR"))),
            anchor_modalities=tuple(str(name) for name in config.get("anchor_modalities", ("Formula", "CNMR"))),
            modality_map=modality_map,
            unit_structure=config.get("unit_structure"),
            slack=float(config.get("slack", 0.10)),
            max_alpha=float(config.get("max_alpha", 0.50)),
            min_alpha=(
                None
                if config.get("min_alpha") is None
                else float(config.get("min_alpha"))
            ),
            distance_temperature=float(config.get("distance_temperature", 0.10)),
        )
        self._tahcd_inference_enabled = bool(config.get("inference_ttt_enabled", False))
        self._tahcd_apply_no_update = bool(config.get("apply_no_update", True))
        self._tahcd_runtime_enabled = True
        # ExposureTTT's block-only recipe freezes every non-exposure module.
        # TAHCD is attached afterwards, so apply that contract explicitly to
        # avoid silently adding fresh trainable parameters to the optimizer.
        if self._exposure_ttt_freeze_backbone:
            for parameter in self.tahcd.parameters():
                parameter.requires_grad_(False)
        print(
            "[tahcd] enabled shared-encoder mode "
            f"d_model={d_model} modalities={self.tahcd.modalities} "
            f"anchors={self.tahcd.anchor_modalities} "
            f"slack={self.tahcd.slack:g}",
            flush=True,
        )

    def _attach_spectral_repair(self) -> None:
        """Attach the hierarchical spectral repair block when requested."""
        config = self.spectral_repair_config
        if not bool(config.get("enabled", False)):
            return
        if self.model_type != CustomModel.__name__:
            raise ValueError("SpectralRepair currently requires model_type=CustomModel")
        d_model = int(getattr(self.hf_model.config, "d_model", 0))
        modality_map = dict(
            config.get(
                "modality_map",
                {name: name for name in ("Formula", "HNMR", "CNMR", "MSMS", "IR")},
            )
        )
        self.spectral_repair = SpectralRepairBlock(
            d_model=d_model,
            modalities=tuple(str(name) for name in config.get("modalities", ("HNMR", "MSMS", "IR"))),
            anchor_modalities=tuple(str(name) for name in config.get("anchor_modalities", ("Formula", "CNMR"))),
            modality_map=modality_map,
            rank=int(config.get("rank", 32)),
            relation_dim=int(config.get("relation_dim", 32)),
            fast_dim=int(config.get("fast_dim", 8)),
            gate_hidden=int(config.get("gate_hidden", 64)),
            min_gate=float(config.get("min_gate", 0.05)),
            max_fast_scale=float(config.get("max_fast_scale", 0.05)),
            instance_conditioned_gate=bool(
                config.get("instance_conditioned_gate", False)
            ),
            gate_init_logit=float(config.get("gate_init_logit", 8.0)),
            max_fast_gate_logit=float(config.get("max_fast_gate_logit", 0.05)),
            max_calibration_ratio=float(config.get("max_calibration_ratio", 0.10)),
            slack_weight=float(config.get("slack_weight", 1.0)),
            clean_weight=float(config.get("clean_weight", 0.05)),
            trust_weight=float(config.get("trust_weight", 0.01)),
            enable_calibration=bool(config.get("enable_calibration", True)),
            enable_gate=bool(config.get("enable_gate", True)),
            enable_relation=bool(config.get("enable_relation", True)),
        )
        stats_path = config.get("source_statistics")
        if stats_path:
            self.spectral_repair.load_source_statistics(str(stats_path))
        self._spectral_repair_runtime_enabled = True
        self._spectral_repair_inference_enabled = bool(
            config.get("inference_tta_enabled", False)
        )
        self._spectral_repair_apply_no_update = bool(
            config.get("apply_no_update", True)
        )
        self._spectral_repair_freeze_backbone = bool(
            config.get("freeze_backbone", False)
        )
        if self._spectral_repair_freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.spectral_repair.parameters():
                parameter.requires_grad_(True)
        print(
            "[spectral-repair] enabled "
            f"d_model={d_model} rank={self.spectral_repair.rank} "
            f"modalities={self.spectral_repair.modalities} "
            f"fast_dim={self.spectral_repair.fast_dim}",
            flush=True,
        )

    def _attach_exposure_ttt(self) -> None:
        """Create the optional pre-fusion ExposureTTT block after base init."""
        config = self.exposure_ttt_config
        if not bool(config.get("enabled", False)):
            return
        if self.model_type != CustomModel.__name__:
            raise ValueError("ExposureTTT currently requires model_type=CustomModel")
        d_model = int(getattr(self.hf_model.config, "d_model", 0))
        modality_map = dict(
            config.get(
                "modality_map",
                {name: name for name in ("Formula", "HNMR", "CNMR", "MSMS", "IR")},
            )
        )
        self.exposure_ttt = ExposureTTTBlock(
            d_model=d_model,
            rank=int(config.get("rank", 8)),
            modalities=tuple(
                str(name)
                for name in config.get("modalities", ("HNMR", "CNMR", "MSMS", "IR"))
            ),
            modality_map=modality_map,
            bound_ratio=float(config.get("bound_ratio", 0.05)),
        )
        self._configure_exposure_trainability()
        print(
            "[exposure-ttt] enabled "
            f"d_model={d_model} rank={self.exposure_ttt.rank} "
            f"modalities={self.exposure_ttt.modalities} "
            f"bound={self.exposure_ttt.bound_ratio:g} "
            f"meta_fraction={self._exposure_ttt_meta_batch_fraction:g}",
            flush=True,
        )

    def _configure_exposure_trainability(self) -> None:
        """Optionally train only ExposureTTT (plus explicitly un-frozen layers).

        Stage B of the recipe starts from a trained source backbone and learns
        the correction basis while keeping the generator and any optional
        fusion/retrieval side modules fixed.  The escape hatch
        ``unfreeze_encoder_layers`` permits a controlled last-layer adaptation
        without silently unfreezing the decoder or embeddings.
        """
        if self.exposure_ttt is None or not self._exposure_ttt_freeze_backbone:
            return
        # Freeze the complete wrapper first.  ``generation_fusion`` and
        # ``multitask_retrieval_heads`` live beside ``hf_model`` and would
        # otherwise remain trainable during the block-only Stage B recipe.
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.exposure_ttt.parameters():
            parameter.requires_grad_(True)

        count = self._exposure_ttt_unfreeze_encoder_layers
        if count == 0:
            return
        encoder = getattr(self.hf_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            raise ValueError(
                "unfreeze_encoder_layers requires a model encoder with a layers sequence"
            )
        if count > len(layers):
            raise ValueError(
                f"unfreeze_encoder_layers={count} exceeds encoder depth {len(layers)}"
            )
        for layer in list(layers)[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        """Keep a frozen source backbone deterministic during Stage B training."""
        super().train(mode)
        exposure_ttt = getattr(self, "exposure_ttt", None)
        freeze_backbone = bool(getattr(self, "_exposure_ttt_freeze_backbone", False))
        if mode and exposure_ttt is not None and freeze_backbone:
            # ``LightningModule.train()`` recursively switches every child to
            # train mode. Frozen dropout in the source generator or optional
            # fusion/retrieval side modules would make the no-update/gain
            # comparison noisy, so restore eval mode for every non-adaptive
            # child and then enable training only on ExposureTTT itself.
            for child in self.children():
                child.eval()
            exposure_ttt.train(True)
            count = int(getattr(self, "_exposure_ttt_unfreeze_encoder_layers", 0))
            if count:
                encoder = getattr(self.hf_model, "encoder", None)
                layers = getattr(encoder, "layers", None)
                if layers is not None:
                    for layer in list(layers)[-count:]:
                        layer.train(True)
        spectral_repair = getattr(self, "spectral_repair", None)
        if mode and spectral_repair is not None and getattr(
            self, "_spectral_repair_freeze_backbone", False
        ):
            for child in self.children():
                child.eval()
            spectral_repair.train(True)
        return self

    def load_compatible_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load checkpoints while allowing the optional ExposureTTT extension.

        A legacy backbone checkpoint may omit every ``exposure_ttt.*`` key and
        is then loaded into the identity-initialized new branch.  Partial new
        branches are rejected so a typo cannot silently produce a hybrid model.
        """
        current_state = self.state_dict()
        current_keys = set(current_state)
        incoming_keys = set(state_dict)
        exposure_prefix = "exposure_ttt."
        current_exposure_keys = {
            key for key in current_keys if key.startswith(exposure_prefix)
        }
        incoming_exposure_keys = {
            key for key in incoming_keys if key.startswith(exposure_prefix)
        }
        tahcd_prefix = "tahcd."
        current_tahcd_keys = {key for key in current_keys if key.startswith(tahcd_prefix)}
        incoming_tahcd_keys = {key for key in incoming_keys if key.startswith(tahcd_prefix)}
        if incoming_tahcd_keys and incoming_tahcd_keys != current_tahcd_keys:
            raise RuntimeError(
                "TAHCD checkpoint keys do not match the configured block; "
                f"missing={sorted(current_tahcd_keys - incoming_tahcd_keys)}, "
                f"unexpected={sorted(incoming_tahcd_keys - current_tahcd_keys)}"
            )
        if incoming_exposure_keys and incoming_exposure_keys != current_exposure_keys:
            raise RuntimeError(
                "ExposureTTT checkpoint keys do not match the configured block; "
                f"missing={sorted(current_exposure_keys - incoming_exposure_keys)}, "
                f"unexpected={sorted(incoming_exposure_keys - current_exposure_keys)}"
            )
        spectral_prefix = "spectral_repair."
        current_spectral_keys = {
            key for key in current_keys if key.startswith(spectral_prefix)
        }
        incoming_spectral_keys = {
            key for key in incoming_keys if key.startswith(spectral_prefix)
        }
        if incoming_spectral_keys and incoming_spectral_keys != current_spectral_keys:
            raise RuntimeError(
                "SpectralRepair checkpoint keys do not match the configured block; "
                f"missing={sorted(current_spectral_keys - incoming_spectral_keys)}, "
                f"unexpected={sorted(incoming_spectral_keys - current_spectral_keys)}"
            )
        allowed_missing = (
            current_exposure_keys if current_exposure_keys and not incoming_exposure_keys else set()
        )
        if current_tahcd_keys and not incoming_tahcd_keys:
            allowed_missing = allowed_missing.union(current_tahcd_keys)
        if current_spectral_keys and not incoming_spectral_keys:
            allowed_missing = allowed_missing.union(current_spectral_keys)
        missing = current_keys - incoming_keys
        unexpected = incoming_keys - current_keys
        non_optional_missing = missing - allowed_missing
        if non_optional_missing or unexpected:
            raise RuntimeError(
                "checkpoint keys do not match the configured model; "
                f"missing={sorted(non_optional_missing)}, unexpected={sorted(unexpected)}"
            )

        merged_state: Dict[str, torch.Tensor] = {}
        expandable_fragments = ("embedding_layer_dict.", "token_ff.weight", "token_ff.bias")
        for key, current_value in current_state.items():
            if key not in state_dict:
                continue
            incoming_value = state_dict[key]
            if incoming_value.shape == current_value.shape:
                merged_state[key] = incoming_value
                continue

            same_trailing_shape = (
                incoming_value.ndim == current_value.ndim
                and incoming_value.shape[1:] == current_value.shape[1:]
            )
            is_row_expansion = (
                same_trailing_shape
                and incoming_value.shape[0] < current_value.shape[0]
                and any(fragment in key for fragment in expandable_fragments)
            )
            if not is_row_expansion:
                raise RuntimeError(
                    f"incompatible checkpoint tensor {key}: "
                    f"checkpoint={tuple(incoming_value.shape)} "
                    f"model={tuple(current_value.shape)}"
                )

            expanded_value = current_value.detach().clone()
            expanded_value[: incoming_value.shape[0]].copy_(incoming_value)
            merged_state[key] = expanded_value

        missing_after, unexpected_after = self.load_state_dict(merged_state, strict=False)
        if set(missing_after) != allowed_missing or unexpected_after:
            raise RuntimeError(
                "checkpoint load produced unexpected key differences; "
                f"missing={sorted(missing_after)}, unexpected={sorted(unexpected_after)}"
            )

    @staticmethod
    def _observed_formulas(batch: Mapping[str, Any]) -> list[str]:
        """Return Formula values supplied as an input modality.

        Formula-guided generation is only valid when Formula is observable at
        inference time.  In particular, never fall back to ``target_smiles``:
        doing so would make validation/test molecular accuracy a target-leaking
        metric.
        """
        values = batch.get("input_formulas")
        if values is None:
            raise ValueError(
                "Formula-guided generation requires observable input_formulas; "
                "the target SMILES cannot be used as a Formula constraint."
            )
        if isinstance(values, str):
            values = [values]
        try:
            formulas = [str(value) for value in values]
        except TypeError as exc:
            raise TypeError("input_formulas must be a sequence of Formula strings.") from exc
        if not formulas:
            raise ValueError("Formula-guided generation received no input Formula values.")
        return formulas

    def _pool_generation_modalities(
        self,
        batch: Dict[str, Any],
        input_ids: Dict[str, Any],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        """Pool each included modality for the generation fusion block."""
        if self.generation_fusion is None:
            return {}, {}, torch.zeros_like(attention_mask, dtype=torch.bool)
        pooled_by_name: Dict[str, list[torch.Tensor]] = {}
        availability_by_name: Dict[str, list[torch.Tensor]] = {}
        token_masks_by_name: Dict[str, list[torch.Tensor]] = {}
        offset = 0
        batch_size = inputs_embeds.shape[0]
        for data_name, modality_input in input_ids.items():
            sequence_length = self._input_sequence_length(modality_input)
            segment = inputs_embeds[:, offset : offset + sequence_length]
            token_mask = attention_mask[:, offset : offset + sequence_length].bool()
            offset += sequence_length

            canonical_name = self.generation_fusion.modality_map.get(data_name)
            if canonical_name not in {
                "Formula",
                "HNMR",
                "CNMR",
                "NMR",
                "MSMS",
                "IR",
            }:
                continue
            configured_availability = None
            batch_availability = batch.get("encoder_modality_availability")
            if isinstance(batch_availability, Mapping):
                configured_availability = batch_availability.get(data_name)
            available = configured_availability
            if available is None:
                available = token_mask.any(dim=1)
            available = available.to(device=segment.device, dtype=torch.bool).reshape(-1)
            if available.numel() != batch_size:
                raise ValueError(
                    f"Availability for {data_name} has {available.numel()} values; "
                    f"expected {batch_size}."
                )
            weights = token_mask.to(dtype=segment.dtype)
            summary = (segment * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            summary = torch.where(available.unsqueeze(-1), summary, torch.zeros_like(summary))
            pooled_by_name.setdefault(canonical_name, []).append(summary)
            availability_by_name.setdefault(canonical_name, []).append(available)
            token_masks_by_name.setdefault(canonical_name, []).append(
                token_mask & available.unsqueeze(-1)
            )

        pooled: Dict[str, torch.Tensor] = {}
        availability: Dict[str, torch.Tensor] = {}
        for name, values in pooled_by_name.items():
            present = torch.stack(availability_by_name[name], dim=0)
            value_stack = torch.stack(values, dim=0)
            if name == "NMR" and value_stack.shape[0] > 1:
                pooled[name] = value_stack.sum(dim=0) / present.to(
                    value_stack.dtype
                ).sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            else:
                pooled[name] = value_stack[0]
            availability[name] = present.any(dim=0)

        target_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        offset = 0
        canonical_segment_masks: Dict[str, torch.Tensor] = {}
        for data_name, modality_input in input_ids.items():
            sequence_length = self._input_sequence_length(modality_input)
            canonical_name = self.generation_fusion.modality_map.get(data_name)
            if canonical_name in token_masks_by_name:
                segment_mask = token_masks_by_name[canonical_name].pop(0)
                canonical_segment_masks.setdefault(
                    canonical_name, torch.zeros_like(attention_mask, dtype=torch.bool)
                )[:, offset : offset + sequence_length] |= segment_mask
            offset += sequence_length

        nmr_mask = canonical_segment_masks.get("NMR", target_mask).clone()
        nmr_mask |= canonical_segment_masks.get("HNMR", target_mask)
        nmr_mask |= canonical_segment_masks.get("CNMR", target_mask)
        formula_mask = canonical_segment_masks.get("Formula", target_mask)
        nmr_available = nmr_mask.any(dim=1)
        formula_available = formula_mask.any(dim=1)
        target_mask = torch.where(nmr_available.unsqueeze(-1), nmr_mask, formula_mask)
        no_anchor = ~(nmr_available | formula_available)
        target_mask = torch.where(no_anchor.unsqueeze(-1), attention_mask.bool(), target_mask)
        return pooled, availability, target_mask

    def _prepare_generation_inputs(
        self,
        batch: Dict[str, Any],
        *,
        apply_modality_dropout: bool = False,
        apply_exposure_block: bool = True,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """Prepare embeddings identically for teacher-forcing and generation."""
        input_ids, attention_mask = self.prepare_encoder_inputs(
            batch, apply_modality_dropout=apply_modality_dropout
        )
        generation_fusion = getattr(self, "generation_fusion", None)
        exposure_ttt = getattr(self, "exposure_ttt", None)
        if generation_fusion is not None and generation_fusion.uses_encoder_token_fusion:
            # READ assumes modality-specific feature encoding. Reuse the same
            # source-trained encoder weights, but reset positions and encode
            # each modality independently before SAF concatenation.
            inputs_embeds = torch.cat(
                [
                    self.multimodal_embedding({name: modality_input})
                    for name, modality_input in input_ids.items()
                ],
                dim=1,
            )
        else:
            inputs_embeds = self.multimodal_embedding(input_ids)
        if generation_fusion is not None and not generation_fusion.uses_encoder_token_fusion:
            pooled, availability, target_mask = self._pool_generation_modalities(
                batch, input_ids, inputs_embeds, attention_mask
            )
            context = generation_fusion(pooled, availability)
            inputs_embeds = inputs_embeds + (
                context.unsqueeze(1) * target_mask.unsqueeze(-1).to(context.dtype)
            )
        if apply_exposure_block and exposure_ttt is not None:
            inputs_embeds = self._apply_exposure_block(
                input_ids,
                inputs_embeds,
                attention_mask,
                fast_state=fast_state,
            )
        return input_ids, attention_mask, inputs_embeds

    def _exposure_modality_spans(
        self,
        input_ids: Mapping[str, Any],
    ) -> tuple[tuple[str, int, int], ...]:
        """Return contiguous spans matching the collator's concatenation order."""
        spans = []
        offset = 0
        for data_name, modality_input in input_ids.items():
            length = self._input_sequence_length(modality_input)
            spans.append((str(data_name), offset, offset + length))
            offset += length
        return tuple(spans)

    def _spectral_fc_reference(
        self,
        input_ids: Mapping[str, Any],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Formula+CNMR alone and return a detached, modality-equal anchor.

        This is a genuinely independent encoder pass: HNMR/MSMS/IR tokens are
        not masked in a full sequence but are absent from the reference input.
        The frozen backbone therefore cannot carry their contamination into the
        conditional relation target.
        """
        block = getattr(self, "spectral_repair", None)
        if block is None:
            raise RuntimeError("SpectralRepair reference requested without a block")
        if hasattr(self.hf_model, "model"):
            encoder = self.hf_model.model.encoder
        else:
            encoder = self.hf_model.encoder

        embed_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        lengths: list[int] = []
        offset = 0
        for data_name, modality_input in input_ids.items():
            length = self._input_sequence_length(modality_input)
            canonical = block.canonical_name(str(data_name))
            if canonical in block.anchor_modalities:
                embed_parts.append(inputs_embeds[:, offset : offset + length])
                mask_parts.append(attention_mask[:, offset : offset + length])
                lengths.append(length)
            offset += length
        if not embed_parts:
            raise ValueError("SpectralRepair requires Formula or CNMR for its FC reference")

        reference_embeds = torch.cat(embed_parts, dim=1)
        reference_mask = torch.cat(mask_parts, dim=1)
        # The reference path is frozen even while the repair block is trained.
        with torch.no_grad():
            reference_output = encoder(
                attention_mask=reference_mask,
                inputs_embeds=reference_embeds,
            )["last_hidden_state"]

        pooled_parts = []
        available_parts = []
        offset = 0
        for length in lengths:
            segment = reference_output[:, offset : offset + length]
            segment_mask = reference_mask[:, offset : offset + length].bool()
            weights = segment_mask.to(segment.dtype).unsqueeze(-1)
            pooled = (segment * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            available = segment_mask.any(dim=1)
            pooled_parts.append(torch.where(available.unsqueeze(-1), pooled, 0.0))
            available_parts.append(available)
            offset += length
        pooled_stack = torch.stack(pooled_parts, dim=0)
        available_stack = torch.stack(available_parts, dim=0)
        counts = available_stack.to(pooled_stack.dtype).sum(dim=0).clamp_min(1.0)
        anchor = pooled_stack.sum(dim=0) / counts.unsqueeze(-1)
        return anchor.detach(), available_stack.any(dim=0)

    def _spectral_unit_ids(
        self,
        input_ids: Mapping[str, Any],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map tokenizer records to shared peak/patch IDs for local gating."""
        block = getattr(self, "spectral_repair", None)
        if block is None:
            raise RuntimeError("SpectralRepair unit IDs requested without a block")
        output = torch.full_like(attention_mask, -1, dtype=torch.long)
        offset = 0
        for data_name, modality_input in input_ids.items():
            length = self._input_sequence_length(modality_input)
            canonical = block.canonical_name(str(data_name))
            if canonical in block.modalities:
                if isinstance(modality_input, Mapping):
                    token_ids = modality_input.get("tokenized_input")
                else:
                    token_ids = modality_input
                if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 2:
                    output[:, offset : offset + length] = block.build_unit_ids(
                        token_ids,
                        attention_mask[:, offset : offset + length].bool(),
                        canonical,
                    )
                else:
                    positions = torch.arange(length, device=attention_mask.device)
                    output[:, offset : offset + length] = torch.where(
                        attention_mask[:, offset : offset + length].bool(),
                        positions.unsqueeze(0),
                        -1,
                    )
            offset += length
        return output

    def _apply_exposure_block(
        self,
        input_ids: Mapping[str, Any],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        exposure_ttt = getattr(self, "exposure_ttt", None)
        if exposure_ttt is None:
            return inputs_embeds
        spans = self._exposure_modality_spans(input_ids)
        return exposure_ttt(
            inputs_embeds,
            attention_mask,
            spans,
            fast_state=fast_state,
        )

    def _encode_generation_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: Optional[Mapping[str, Any]] = None,
        tahcd_state: Optional[Mapping[str, torch.Tensor]] = None,
        apply_tahcd: bool = True,
        spectral_state: Optional[Mapping[str, torch.Tensor]] = None,
        apply_spectral_repair: bool = True,
        spectral_anchor: Optional[torch.Tensor] = None,
        spectral_anchor_available: Optional[torch.Tensor] = None,
        spectral_unit_ids: Optional[torch.Tensor] = None,
    ) -> Any:
        """Encode each modality and apply token fusion after the shared backbone."""
        if hasattr(self.hf_model, "model"):
            encoder = self.hf_model.model.encoder
        else:
            encoder = self.hf_model.encoder
        generation_fusion = getattr(self, "generation_fusion", None)
        spectral_repair = getattr(self, "spectral_repair", None)
        spectral_encoder_bias = None
        spectral_decoder_bias = None
        spectral_diagnostics: dict[str, Any] = {}
        if (
            apply_spectral_repair
            and spectral_repair is not None
            and getattr(self, "_spectral_repair_runtime_enabled", True)
        ):
            if input_ids is None:
                raise ValueError("SpectralRepair requires modality input spans")
            spectral_spans = self._exposure_modality_spans(input_ids)
            if spectral_anchor is None:
                spectral_anchor, spectral_anchor_available = self._spectral_fc_reference(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                )
            if spectral_unit_ids is None:
                spectral_unit_ids = self._spectral_unit_ids(input_ids, attention_mask)
            inputs_embeds, spectral_encoder_bias, spectral_decoder_bias, spectral_diagnostics = spectral_repair.apply(
                inputs_embeds,
                attention_mask,
                spectral_spans,
                fast_state=spectral_state,
                anchor=spectral_anchor,
                anchor_available=spectral_anchor_available,
                unit_ids=spectral_unit_ids,
                bypass=False,
                return_diagnostics=True,
            )
        uses_read_fusion = (
            generation_fusion is not None
            and generation_fusion.uses_encoder_token_fusion
        )
        if uses_read_fusion:
            if input_ids is None:
                raise ValueError("Encoder-token fusion requires modality input spans.")
            encoded_parts = []
            modality_id_parts = []
            encoder_outputs = None
            offset = 0
            for data_name, modality_input in input_ids.items():
                sequence_length = self._input_sequence_length(modality_input)
                segment_output = encoder(
                    attention_mask=attention_mask[:, offset : offset + sequence_length],
                    inputs_embeds=inputs_embeds[:, offset : offset + sequence_length],
                    key_attention_bias=(
                        spectral_encoder_bias[:, offset : offset + sequence_length]
                        if spectral_encoder_bias is not None
                        else None
                    ),
                )
                encoded_parts.append(segment_output["last_hidden_state"])
                canonical_name = generation_fusion.modality_map.get(
                    data_name, data_name
                )
                if canonical_name in {"HNMR", "CNMR"}:
                    canonical_name = "NMR"
                modality_index = {
                    "Formula": 0,
                    "NMR": 1,
                    "MSMS": 2,
                    "IR": 3,
                }.get(canonical_name, 1)
                modality_id_parts.append(
                    torch.full(
                        (inputs_embeds.shape[0], sequence_length),
                        modality_index,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
                if encoder_outputs is None:
                    encoder_outputs = segment_output
                offset += sequence_length
            if encoder_outputs is None or offset != attention_mask.shape[1]:
                raise ValueError("Encoder-token fusion requires a complete modality span.")
            encoder_outputs["last_hidden_state"] = torch.cat(encoded_parts, dim=1)
            modality_ids = torch.cat(modality_id_parts, dim=1)
            if "attention_mask" in encoder_outputs:
                encoder_outputs["attention_mask"] = attention_mask
            key_token_counts = self._read_key_token_counts(
                input_ids,
                attention_mask,
            )
            encoder_outputs["key_token_counts"] = key_token_counts
            encoder_outputs["last_hidden_state"] = generation_fusion.forward_encoder(
                encoder_outputs["last_hidden_state"],
                attention_mask,
                key_token_counts=key_token_counts,
                modality_ids=modality_ids,
            )
        else:
            encoder_outputs = encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                key_attention_bias=spectral_encoder_bias,
            )
        if spectral_decoder_bias is not None:
            encoder_outputs["key_attention_bias"] = spectral_decoder_bias
            encoder_outputs["spectral_diagnostics"] = spectral_diagnostics
        if (
            apply_tahcd
            and getattr(self, "_tahcd_runtime_enabled", True)
            and getattr(self, "tahcd", None) is not None
        ):
            if input_ids is None:
                raise ValueError("TAHCD requires modality input spans")
            encoder_outputs = self._apply_tahcd_to_encoder_outputs(
                encoder_outputs,
                input_ids,
                attention_mask,
                fast_state=tahcd_state,
            )
        return encoder_outputs

    def _apply_tahcd_to_encoder_outputs(
        self,
        encoder_outputs: Any,
        input_ids: Mapping[str, Any],
        attention_mask: torch.Tensor,
        *,
        fast_state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Any:
        tahcd = getattr(self, "tahcd", None)
        if tahcd is None or not getattr(self, "_tahcd_runtime_enabled", True):
            return encoder_outputs
        spans = self._exposure_modality_spans(input_ids)
        hidden, diagnostics = tahcd(
            encoder_outputs["last_hidden_state"],
            attention_mask,
            spans,
            fast_state=fast_state,
            return_diagnostics=True,
        )
        encoder_outputs["last_hidden_state"] = hidden
        encoder_outputs["tahcd_diagnostics"] = diagnostics
        return encoder_outputs

    def _read_key_token_counts(
        self,
        input_ids: Optional[Mapping[str, Any]],
        attention_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Assign each token the valid-token count of its input modality span."""
        if input_ids is None:
            return None
        counts = torch.ones_like(attention_mask, dtype=torch.float32)
        valid = attention_mask.to(dtype=torch.bool)
        spans: list[tuple[int, int, str]] = []
        group_counts: Dict[str, torch.Tensor] = {}
        offset = 0
        for data_name, modality_input in input_ids.items():
            sequence_length = self._input_sequence_length(modality_input)
            segment_valid = valid[:, offset : offset + sequence_length]
            canonical_name = self.generation_fusion.modality_map.get(
                data_name,
                data_name,
            )
            if canonical_name in {"HNMR", "CNMR"}:
                canonical_name = "NMR"
            segment_count = segment_valid.sum(dim=1, keepdim=True)
            if canonical_name in group_counts:
                group_counts[canonical_name] = (
                    group_counts[canonical_name] + segment_count
                )
            else:
                group_counts[canonical_name] = segment_count
            spans.append((offset, sequence_length, canonical_name))
            offset += sequence_length
        if offset != attention_mask.shape[1]:
            raise ValueError(
                "READ modality spans do not cover the concatenated encoder sequence."
            )
        for span_offset, sequence_length, canonical_name in spans:
            counts[:, span_offset : span_offset + sequence_length] = group_counts[
                canonical_name
            ].clamp_min(1)
        return counts

    def configure_optimizers(self):
        """Set up optimisers for pytorch lightning"""
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters remain after ExposureTTT freezing.")

        optim = OPTIMISER_REGISTRY[self.optimiser](
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
        )

        if self.lr_scheduler == "cyclic":
            print("Using cyclical LR scheduler")
            cycle_sch = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}
            return [optim], [sch]
        
        elif self.lr_scheduler == "exp":
            print("Using exponential LR scheduler")
            exp_sch = ExponentialLR(optim, gamma=self.gamma)
            sch = {"scheduler": exp_sch, "interval": "epoch"}
            return [optim], [sch]
        
        elif self.lr_scheduler == "constant":
            print("Using constant LR")
            return [optim]

    @staticmethod
    def _input_sequence_length(modality_input: Any) -> int:
        """Return the sequence length of a batch-first modality input."""
        if isinstance(modality_input, torch.Tensor):
            if modality_input.ndim < 2:
                raise ValueError("Encoder modality inputs must have a sequence dimension.")
            return int(modality_input.shape[1])
        if isinstance(modality_input, Mapping):
            primary_input = modality_input.get("tokenized_input")
            if primary_input is None:
                primary_input = next(
                    (value for value in modality_input.values() if isinstance(value, torch.Tensor)),
                    None,
                )
            if primary_input is None:
                raise ValueError("Nested encoder modality input contains no tensor.")
            return HFWrapper._input_sequence_length(primary_input)
        raise TypeError(
            f"Unsupported encoder modality input type: {type(modality_input).__name__}"
        )

    def prepare_encoder_inputs(
        self,
        batch: Dict[str, Any],
        *,
        apply_modality_dropout: bool = False,
    ) -> tuple[Dict[str, Any], torch.Tensor]:
        """Build batch-first encoder inputs after deterministic modality exclusion."""
        input_ids = {
            modality: _to_batch_first(modality_input)
            for modality, modality_input in batch["encoder_input"].items()
        }
        modalities_to_drop = set(self.excluded_input_modalities)

        if (
            apply_modality_dropout
            and isinstance(self.modality_dropout, ListConfig)
            and self.training
        ):
            dropout_candidates = [
                modality
                for modality in self.modality_dropout
                if modality not in modalities_to_drop
            ]
            if dropout_candidates:
                selected = np.random.choice(
                    dropout_candidates,
                    np.random.randint(0, len(dropout_candidates)),
                    replace=False,
                )
                modalities_to_drop.update(str(modality) for modality in selected)
        elif (
            apply_modality_dropout
            and isinstance(self.modality_dropout, DictConfig)
            and self.training
        ):
            modalities_to_drop.update(
                _sample_grouped_modality_subset(
                    self.modality_dropout,
                    set(input_ids).difference(modalities_to_drop),
                )
            )

        included_modalities = [
            modality for modality in input_ids if modality not in modalities_to_drop
        ]
        if not included_modalities:
            raise ValueError("All encoder input modalities were excluded.")

        modality_pad_masks = batch.get("encoder_modality_pad_masks")
        if isinstance(modality_pad_masks, Mapping) and all(
            modality in modality_pad_masks for modality in included_modalities
        ):
            attention_mask = torch.cat(
                [
                    (~modality_pad_masks[modality]).int().T
                    for modality in included_modalities
                ],
                dim=-1,
            )
        else:
            full_attention_mask = (~batch["encoder_pad_mask"]).int().T
            attention_mask_parts = []
            offset = 0
            for modality, modality_input in input_ids.items():
                sequence_length = self._input_sequence_length(modality_input)
                if modality in included_modalities:
                    attention_mask_parts.append(
                        full_attention_mask[:, offset : offset + sequence_length]
                    )
                offset += sequence_length
            attention_mask = torch.cat(attention_mask_parts, dim=-1)

        filtered_input_ids = {
            modality: input_ids[modality] for modality in included_modalities
        }
        return filtered_input_ids, attention_mask

    def _forward_prepared(
        self,
        batch: Dict[str, Any],
        input_ids: Mapping[str, Any],
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        spectral_state: Optional[Mapping[str, torch.Tensor]] = None,
        apply_spectral_repair: bool = True,
        spectral_anchor: Optional[torch.Tensor] = None,
        spectral_anchor_available: Optional[torch.Tensor] = None,
        spectral_unit_ids: Optional[torch.Tensor] = None,
    ) -> Seq2SeqModelOutput:
        """Run the decoder on already prepared encoder inputs.

        Keeping preparation separate is required by ExposureTTT: the inner
        exposure views reuse one collated batch while the outer losses use the
        same full view with two different fast states.
        """
        decoder_input = batch["decoder_input"][self.target_modality].transpose(1, 0)

        decoder_attention_mask = (~batch["decoder_pad_mask"]).int().T

        labels = batch["target"].T.contiguous()

        # Keep the collated target immutable across QMF's two forwards.
        labels = labels.masked_fill(
            labels.eq(self.target_tokenizer.pad_token_id),
            -100,
        )

        kwargs = {}
        if isinstance(self.hf_model, CustomModel) and "encoder_alignment_input" in batch:
            kwargs = {"encoder_align_target": batch["encoder_alignment_input"]}
        
        use_encoder_outputs = (
            self.generation_fusion is not None
            and self.generation_fusion.uses_encoder_token_fusion
        ) or (
            getattr(self, "tahcd", None) is not None
            and getattr(self, "_tahcd_runtime_enabled", True)
        ) or (
            getattr(self, "spectral_repair", None) is not None
            and getattr(self, "_spectral_repair_runtime_enabled", True)
        )
        if use_encoder_outputs:
            encoder_outputs = self._encode_generation_inputs(
                inputs_embeds,
                attention_mask,
                input_ids,
                spectral_state=spectral_state,
                apply_spectral_repair=apply_spectral_repair,
                spectral_anchor=spectral_anchor,
                spectral_anchor_available=spectral_anchor_available,
                spectral_unit_ids=spectral_unit_ids,
            )
            model_output = self.hf_model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )
        else:
            model_output = self.hf_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )

        if self.multitask_retrieval_heads is not None and "retrieval_targets" in batch:
            retrieval_predictions = self.predict_multitask_retrieval(batch)
            targets = batch["retrieval_targets"]
            fingerprint_target = targets["fingerprint"]
            task_targets = {name: value for name, value in targets.items() if name != "fingerprint"}
            retrieval_config = self.multitask_retrieval_config or {}
            retrieval_loss, retrieval_losses = self.multitask_retrieval_heads.loss(
                retrieval_predictions,
                fingerprint_target,
                task_targets,
                fused_fingerprint_weight=float(
                    retrieval_config.get("fused_fingerprint_loss_weight", 1.0)
                ),
                modality_fingerprint_weight=float(
                    retrieval_config.get("modality_fingerprint_loss_weight", 0.5)
                ),
                task_weight=float(retrieval_config.get("task_loss_weight", 1.0)),
                alignment_weight=float(retrieval_config.get("alignment_loss_weight", 0.1)),
            )
            retrieval_scale = float(retrieval_config.get("loss_weight", 1.0))
            model_output.loss = model_output.loss + retrieval_scale * retrieval_loss
            if isinstance(model_output, CustomLMOutput) and model_output.loss_dict is not None:
                model_output.loss_dict.update(
                    {f"retrieval_{name}": value for name, value in retrieval_losses.items()}
                )
            model_output["retrieval_predictions"] = retrieval_predictions

        return model_output

    def forward(self, batch: Dict[str, Any]) -> Seq2SeqModelOutput:
        """Forward step with the normal source-side modality dropout policy."""
        input_ids, attention_mask, inputs_embeds = self._prepare_generation_inputs(
            batch,
            apply_modality_dropout=True,
            apply_exposure_block=True,
        )
        return self._forward_prepared(batch, input_ids, attention_mask, inputs_embeds)

    def _exposure_available_modalities(
        self,
        attention_mask: torch.Tensor,
        spans: Sequence[tuple[str, int, int]],
    ) -> list[str]:
        """Return configured spectroscopy modalities present in at least one row."""
        if self.exposure_ttt is None:
            return []
        available: list[str] = []
        for canonical in self.exposure_ttt.modalities:
            if any(
                self.exposure_ttt._canonical_name(data_name) == canonical
                and attention_mask[:, start:end].bool().any()
                for data_name, start, end in spans
            ):
                available.append(canonical)
        return available

    def _select_exposure_modality(
        self,
        attention_mask: torch.Tensor,
        spans: Sequence[tuple[str, int, int]],
        *,
        requested: Optional[str] = None,
        deterministic: bool = False,
    ) -> Optional[str]:
        available = self._exposure_available_modalities(attention_mask, spans)
        if not available:
            return None
        if requested is not None:
            canonical = self.exposure_ttt._canonical_name(requested)  # type: ignore[union-attr]
            if canonical not in available:
                raise ValueError(
                    f"Requested ExposureTTT modality {requested!r} is unavailable; "
                    f"available={available}"
                )
            return canonical
        if deterministic:
            return available[0]
        return str(np.random.choice(available))

    def _exposure_inner_update(
        self,
        input_ids: Mapping[str, Any],
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        fast_state: Mapping[str, torch.Tensor],
        *,
        requested_modality: Optional[str] = None,
        deterministic: bool = False,
        shuffle_exposure: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | str]]:
        """Compute one label-free exposure update without mutating model weights.

        ``shuffle_exposure`` is reserved for the validation negative control:
        it swaps only the selected modality's exposure view across batch items.
        The full view remains molecule-correct, and no target labels enter the
        inner objective in either condition.
        """
        block = self.exposure_ttt
        if block is None:
            return dict(fast_state), {}
        spans = self._exposure_modality_spans(input_ids)
        selected = self._select_exposure_modality(
            attention_mask,
            spans,
            requested=requested_modality,
            deterministic=deterministic,
        )
        if selected is None:
            zero = inputs_embeds.new_zeros(())
            return dict(fast_state), {
                "inner_loss": zero,
                "exposure_loss": zero,
                "drop_loss": zero,
                "fast_grad_norm": zero,
                "selected_modality": "none",
            }

        ratio_key = "inference_q_ratios" if deterministic else "q_ratios"
        ratios = tuple(
            float(value)
            for value in self.exposure_ttt_config.get(
                ratio_key,
                self.exposure_ttt_config.get("q_ratios", (0.25, 0.5, 1.0)),
            )
        )
        view_masks = block.build_exposure_masks(
            attention_mask,
            spans,
            selected,
            ratios=ratios,
            deterministic=deterministic,
        )
        if not view_masks:
            zero = inputs_embeds.new_zeros(())
            return dict(fast_state), {
                "inner_loss": zero,
                "exposure_loss": zero,
                "drop_loss": zero,
                "fast_grad_norm": zero,
                "selected_modality": selected,
            }

        # Each view is encoded separately, but the fast state is shared.  The
        # views carry only spectra-derived masks; no target SMILES or Formula
        # labels enter this inner objective.
        # Detach the source embedding graph for the first-order inner pass.  It
        # prevents ``autograd.grad(..., retain_graph=False)`` from freeing the
        # embedding graph later needed by the outer no-update/update forwards.
        inner_inputs_embeds = inputs_embeds.detach()
        view_names: list[str] = []
        representations: list[torch.Tensor] = []
        full_mask = attention_mask.bool()
        full_embeds = block(
            inner_inputs_embeds,
            full_mask,
            spans,
            fast_state=fast_state,
        )
        full_encoded = self._encode_generation_inputs(
            full_embeds,
            full_mask,
            input_ids,
        )
        full_rep = block.masked_pool(
            full_encoded["last_hidden_state"],
            full_mask,
            eps=block.eps,
        )
        view_names.append("full")
        representations.append(full_rep)
        shift_config = self.exposure_ttt_config.get("shift_augmentation", {})
        if not isinstance(shift_config, Mapping):
            raise TypeError("exposure_ttt_config.shift_augmentation must be a mapping")
        for name, view_mask in view_masks:
            shifted_inputs, shifted_mask = block.apply_shift_augmentation(
                inner_inputs_embeds,
                view_mask,
                spans,
                selected,
                enabled=bool(shift_config.get("enabled", False)),
                scale_jitter=float(shift_config.get("scale_jitter", 0.0)),
                noise_std=float(shift_config.get("noise_std", 0.0)),
                token_dropout=float(shift_config.get("token_dropout", 0.0)),
                baseline_shift=float(shift_config.get("baseline_shift", 0.0)),
                deterministic=deterministic,
            )
            if shuffle_exposure:
                shifted_inputs, shifted_mask = block.shuffle_modality_view(
                    shifted_inputs,
                    shifted_mask,
                    spans,
                    selected,
                    deterministic=deterministic,
                )
            view_embeds = block(
                shifted_inputs,
                shifted_mask,
                spans,
                fast_state=fast_state,
            )
            encoded = self._encode_generation_inputs(
                view_embeds,
                shifted_mask,
                input_ids,
            )
            representations.append(
                block.masked_pool(
                    encoded["last_hidden_state"],
                    shifted_mask,
                    eps=block.eps,
                )
            )
            view_names.append(name)

        exposure_loss = block.consistency_loss(representations, eps=block.eps)
        drop_mask = block.build_drop_mask(attention_mask, spans, selected)
        if drop_mask is None:
            drop_loss = exposure_loss * 0.0
        else:
            shifted_drop_inputs, shifted_drop_mask = block.apply_shift_augmentation(
                inner_inputs_embeds,
                drop_mask,
                spans,
                selected,
                enabled=bool(shift_config.get("enabled", False)),
                scale_jitter=float(shift_config.get("scale_jitter", 0.0)),
                noise_std=float(shift_config.get("noise_std", 0.0)),
                token_dropout=float(shift_config.get("token_dropout", 0.0)),
                baseline_shift=float(shift_config.get("baseline_shift", 0.0)),
                deterministic=deterministic,
            )
            if shuffle_exposure:
                shifted_drop_inputs, shifted_drop_mask = block.shuffle_modality_view(
                    shifted_drop_inputs,
                    shifted_drop_mask,
                    spans,
                    selected,
                    deterministic=deterministic,
                )
            drop_embeds = block(
                shifted_drop_inputs,
                shifted_drop_mask,
                spans,
                fast_state=fast_state,
            )
            drop_encoded = self._encode_generation_inputs(
                drop_embeds,
                shifted_drop_mask,
                input_ids,
            )
            drop_rep = block.masked_pool(
                drop_encoded["last_hidden_state"],
                shifted_drop_mask,
                eps=block.eps,
            )
            drop_loss = 1.0 - functional.cosine_similarity(
                drop_rep,
                full_rep.detach(),
                dim=-1,
                eps=block.eps,
            ).mean()

        trust = inputs_embeds.new_zeros(())
        for name in block.modalities:
            trust = trust + (fast_state[name] - block.a_meta[name]).pow(2).mean()
        inner_loss = (
            float(self.exposure_ttt_config.get("exposure_loss_weight", 1.0))
            * exposure_loss
            + float(self.exposure_ttt_config.get("drop_loss_weight", 0.5))
            * drop_loss
            + float(self.exposure_ttt_config.get("trust_weight", 0.01)) * trust
        )

        state_values = tuple(fast_state[name] for name in block.modalities)
        if inner_loss.requires_grad:
            gradients = torch.autograd.grad(
                inner_loss,
                state_values,
                create_graph=bool(
                    self.exposure_ttt_config.get("create_graph", False)
                ),
                retain_graph=False,
                allow_unused=True,
            )
        else:
            gradients = tuple(None for _ in state_values)
        step_size = float(self.exposure_ttt_config.get("adaptation_lr", 0.1))
        step_scales = dict(
            self.exposure_ttt_config.get(
                "modality_step_scales",
                {"HNMR": 1.0, "CNMR": 0.25, "MSMS": 1.0, "IR": 1.0},
            )
        )
        updated_state: dict[str, torch.Tensor] = {}
        selected_gradient_norm = inputs_embeds.new_zeros(())
        for name, gradient in zip(block.modalities, gradients):
            value = fast_state[name]
            if name == selected and gradient is not None:
                gradient = gradient.detach()
                clip = float(self.exposure_ttt_config.get("fast_grad_clip", 0.0))
                if clip > 0.0:
                    norm = gradient.float().norm(dim=-1, keepdim=True).clamp_min(block.eps)
                    gradient = gradient * (clip / norm).clamp(max=1.0).to(gradient.dtype)
                selected_gradient_norm = gradient.float().norm(dim=-1).mean().to(
                    dtype=inputs_embeds.dtype
                )
                value = value - step_size * float(step_scales.get(name, 1.0)) * gradient
            updated_state[name] = value
        return updated_state, {
            "inner_loss": inner_loss.detach(),
            "exposure_loss": exposure_loss.detach(),
            "drop_loss": drop_loss.detach(),
            "fast_grad_norm": selected_gradient_norm.detach(),
            "selected_modality": selected,
            "shuffle_exposure": bool(shuffle_exposure),
        }

    def forward_exposure_ttt(
        self,
        batch: Dict[str, Any],
        *,
        requested_modality: Optional[str] = None,
        deterministic: bool = False,
        shuffle_exposure: bool = False,
        no_grad_outer: bool = False,
    ) -> tuple[Seq2SeqModelOutput, dict[str, Any]]:
        """Run the first-order ExposureTTT outer objective on a full batch.

        The optional shuffled mode is a negative control only; it never changes
        the full-view outer input or the labels used for the generation loss.
        During validation, the outer losses are only reported and do not need an
        autograd graph.  ``no_grad_outer`` therefore releases the large encoder
        and decoder activation graph while retaining gradients for the small,
        label-free inner fast-state update above.
        """
        if self.exposure_ttt is None:
            output = self.forward(batch)
            return output, {"outer_loss": output.loss}
        input_ids, attention_mask, raw_embeds = self._prepare_generation_inputs(
            batch,
            apply_modality_dropout=False,
            apply_exposure_block=False,
        )
        fast_state = self.exposure_ttt.initial_fast_state(
            raw_embeds.shape[0],
            detach=False,
            device=raw_embeds.device,
            dtype=raw_embeds.dtype,
        )
        updated_state, inner_diagnostics = self._exposure_inner_update(
            input_ids,
            raw_embeds,
            attention_mask,
            fast_state,
            requested_modality=requested_modality,
            deterministic=deterministic,
            shuffle_exposure=shuffle_exposure,
        )
        zero_embeds = self._apply_exposure_block(
            input_ids,
            raw_embeds,
            attention_mask,
            fast_state=fast_state,
        )
        plus_embeds = self._apply_exposure_block(
            input_ids,
            raw_embeds,
            attention_mask,
            fast_state=updated_state,
        )
        if no_grad_outer:
            with torch.no_grad():
                zero_output = self._forward_prepared(
                    batch,
                    input_ids,
                    attention_mask,
                    zero_embeds,
                )
                plus_output = self._forward_prepared(
                    batch,
                    input_ids,
                    attention_mask,
                    plus_embeds,
                )
        else:
            zero_output = self._forward_prepared(
                batch,
                input_ids,
                attention_mask,
                zero_embeds,
            )
            plus_output = self._forward_prepared(
                batch,
                input_ids,
                attention_mask,
                plus_embeds,
            )
        zero_loss = zero_output.loss
        plus_loss = plus_output.loss
        if zero_loss is None or plus_loss is None:
            raise RuntimeError("ExposureTTT outer loss requires teacher-forced labels")
        gain = functional.relu(
            plus_loss
            - zero_loss.detach()
            + float(self.exposure_ttt_config.get("gain_margin", 0.0))
        )
        outer_loss = (
            plus_loss
            + float(self.exposure_ttt_config.get("zero_loss_weight", 0.5)) * zero_loss
            + float(self.exposure_ttt_config.get("gain_loss_weight", 1.0)) * gain
        )
        inner_diagnostics = dict(inner_diagnostics)
        inner_diagnostics.update(
            {
                "zero_output": zero_output,
                "zero_loss": zero_loss.detach(),
                "plus_loss": plus_loss.detach(),
                "gain_loss": gain.detach(),
                "outer_loss": outer_loss,
                "updated_state": updated_state,
            }
        )
        return plus_output, inner_diagnostics

    def encode_modalities_for_retrieval(
        self,
        batch: Dict[str, Any],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode each spectroscopy modality separately with the shared encoder."""
        if self.multitask_retrieval_heads is None:
            raise RuntimeError("multitask_retrieval_config is required for retrieval encoding.")
        if not hasattr(self.hf_model, "encode"):
            raise TypeError("Task-specialized retrieval currently requires CustomModel.encode().")
        if "encoder_modality_pad_masks" not in batch:
            raise KeyError("The batch does not contain per-modality pad masks.")

        retrieval_config = self.multitask_retrieval_config or {}
        modality_map = dict(
            retrieval_config.get(
                "modality_map",
                {"MSMS": "MSMS", "HNMR": "Multiplets", "CNMR": "Carbon", "IR": "IR"},
            )
        )
        representations: Dict[str, torch.Tensor] = {}
        availability: Dict[str, torch.Tensor] = {}
        for canonical_name, data_name in modality_map.items():
            if (
                data_name in self.excluded_input_modalities
                or data_name not in batch["encoder_input"]
            ):
                continue
            modality_input = batch["encoder_input"][data_name]
            input_ids = _to_batch_first(modality_input)
            inputs_embeds = self.multimodal_embedding({data_name: input_ids})
            attention_mask = (~batch["encoder_modality_pad_masks"][data_name]).int().T
            encoder_outputs = self.hf_model.encode(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            hidden = encoder_outputs["last_hidden_state"]
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            representations[canonical_name] = (hidden * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1.0)
            if "encoder_modality_availability" in batch:
                availability[canonical_name] = batch[
                    "encoder_modality_availability"
                ][data_name].to(device=hidden.device, dtype=torch.bool)
            else:
                availability[canonical_name] = attention_mask.bool().any(dim=1)
        return representations, availability

    def predict_multitask_retrieval(
        self,
        batch: Dict[str, Any],
        reliability: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        """Predict fused fingerprints and modality-specific retrieval tasks."""
        if self.multitask_retrieval_heads is None:
            raise RuntimeError("multitask_retrieval_config is required.")
        representations, availability = self.encode_modalities_for_retrieval(batch)
        return self.multitask_retrieval_heads(
            representations,
            reliability=reliability,
            availability=availability,
        )

    def _default_generation_use_cache(self) -> bool:
        """Return a decoder-cache setting safe for the wrapped model."""
        # CustomModel accepts the HF flag for API compatibility but its
        # decoder intentionally recomputes the full prefix every step.
        if isinstance(
            self.hf_model,
            (CustomModel, CustomBartForConditionalGeneration),
        ):
            return False
        return bool(getattr(getattr(self.hf_model, "config", None), "use_cache", True))

    def generate(
        self,
        batch: Dict[str, Any],
        n_beams: int = 1,
        logits_processor: Optional[LogitsProcessor] = None,
        generation_state: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
        inner_update_enabled: Optional[bool] = None,
        exposure_block_enabled: bool = True,
        shuffle_exposure: bool = False,
    ) -> torch.Tensor:
        """Adapter for HF .generate.
        Args:
            batch: batch containing input, mask, etc.
            n_beams: How many beams are used for beam search
            generation_state: Optional encoder output prepared for this batch.
                Reusing it avoids repeating the multimodal encoder when an SGEM
                sample needs multiple decoder-only passes.
            use_cache: Override decoder KV-cache usage.  The custom Transformer
                decoder does not implement past-key-value reuse, so it remains
                disabled there; standard Hugging Face decoders follow their
                configuration default.
        Returns:
            torch.Tensor: generated sequences
        """

        if generation_state is None:
            generation_state = self.prepare_generation_state(
                batch,
                inner_update_enabled=inner_update_enabled,
                exposure_block_enabled=exposure_block_enabled,
                shuffle_exposure=shuffle_exposure,
            )
        attention_mask = generation_state["attention_mask"]
        encoder_outputs = generation_state["encoder_outputs"]

        if use_cache is None:
            use_cache = self._default_generation_use_cache()

        # Transformers expands encoder outputs in place during beam search.
        # SGEM reuses one autograd-connected state for beam-10, beam-5, and
        # teacher-forced scoring, so isolate each generation call's expansion
        # to a shallow ModelOutput copy while retaining the original tensors.
        generation_encoder_outputs = copy.copy(encoder_outputs)
        generated_sequences = self.hf_model.generate(
            encoder_outputs=generation_encoder_outputs,
            attention_mask=attention_mask,
            num_beams=n_beams,
            num_return_sequences=n_beams,
            generation_config=self.generation_config,
            logits_processor=logits_processor,
            use_cache=bool(use_cache),
        )

        return generated_sequences

    def _tahcd_pseudo_reconstruction_loss(
        self,
        encoder_outputs: Mapping[str, Any],
        attention_mask: torch.Tensor,
        pseudo_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-force frozen generated text for label-free TTCE."""
        if pseudo_sequences.ndim != 2 or pseudo_sequences.shape[1] < 2:
            raise ValueError("pseudo_sequences must have shape [batch, length>=2]")
        decoder_input = pseudo_sequences[:, :-1].contiguous()
        next_tokens = pseudo_sequences[:, 1:].contiguous()
        decoder_attention_mask = decoder_input.ne(self.target_tokenizer.pad_token_id).long()
        output = self.hf_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_attention_mask,
            labels=None,
            use_cache=False,
        )
        token_loss = functional.cross_entropy(
            output.logits.float().reshape(-1, output.logits.shape[-1]),
            next_tokens.reshape(-1),
            ignore_index=self.target_tokenizer.pad_token_id,
            reduction="none",
        ).view_as(next_tokens)
        valid = next_tokens.ne(self.target_tokenizer.pad_token_id)
        return token_loss.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1)

    def prepare_generation_state(
        self,
        batch: Dict[str, Any],
        *,
        inner_update_enabled: Optional[bool] = None,
        exposure_block_enabled: bool = True,
        shuffle_exposure: bool = False,
    ) -> Dict[str, Any]:
        """Prepare encoder outputs once for repeated decoder passes.

        The returned state intentionally retains the encoder autograd graph. SGEM
        uses the same state for its no-grad beam searches and its subsequent
        teacher-forced gradient pass, so the encoder-side TTA parameters remain
        connected to the loss.
        """
        input_ids, attention_mask, inputs_embeds = self._prepare_generation_inputs(
            batch,
            apply_modality_dropout=False,
            apply_exposure_block=False,
        )
        diagnostics: dict[str, Any] = {}
        exposure_ttt = getattr(self, "exposure_ttt", None)
        if exposure_ttt is not None and exposure_block_enabled:
            if inner_update_enabled is None:
                inner_update_enabled = bool(
                    getattr(self, "_exposure_ttt_inference_enabled", False)
                )
            fast_state = exposure_ttt.initial_fast_state(
                inputs_embeds.shape[0],
                detach=True,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            if bool(inner_update_enabled):
                # Evaluation is normally wrapped in torch.no_grad().  The
                # label-free inner objective explicitly re-enables gradients;
                # the resulting state is detached before decoder generation.
                with torch.enable_grad():
                    fast_state, diagnostics = self._exposure_inner_update(
                        input_ids,
                        inputs_embeds,
                        attention_mask,
                        {
                            name: value.detach().clone().requires_grad_(True)
                            for name, value in fast_state.items()
                        },
                        requested_modality=self.exposure_ttt_config.get(
                            "inference_modality"
                        ),
                        deterministic=True,
                        shuffle_exposure=shuffle_exposure,
                    )
                inputs_embeds = self._apply_exposure_block(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    fast_state={
                        name: value.detach() for name, value in fast_state.items()
                    },
                )
            else:
                inputs_embeds = self._apply_exposure_block(
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    fast_state=None,
                )
        else:
            diagnostics = {}
        spectral_repair = getattr(self, "spectral_repair", None)
        spectral_state = None
        spectral_anchor = None
        spectral_anchor_available = None
        spectral_unit_ids = None
        spectral_diagnostics: dict[str, Any] = {}
        spectral_active = False
        if (
            spectral_repair is not None
            and getattr(self, "_spectral_repair_runtime_enabled", True)
        ):
            update_enabled = bool(
                getattr(self, "_spectral_repair_inference_enabled", False)
            )
            spectral_active = update_enabled or bool(
                getattr(self, "_spectral_repair_apply_no_update", True)
            )
        if spectral_repair is not None and spectral_active:
            spans = self._exposure_modality_spans(input_ids)
            spectral_anchor, spectral_anchor_available = self._spectral_fc_reference(
                input_ids,
                inputs_embeds,
                attention_mask,
            )
            spectral_unit_ids = self._spectral_unit_ids(input_ids, attention_mask)
            spectral_state = spectral_repair.initial_fast_state(
                inputs_embeds.shape[0],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                requires_grad=False,
            )
            if update_enabled:
                repair_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                repair_names = set(spectral_repair.modalities)
                for data_name, start, end in spans:
                    if spectral_repair.canonical_name(data_name) in repair_names:
                        repair_mask[:, start:end] = attention_mask[:, start:end].bool()
                with torch.no_grad():
                    (
                        _baseline_repaired,
                        baseline_attention_bias,
                        _baseline_decoder_bias,
                        baseline_repair_diagnostics,
                    ) = spectral_repair.apply(
                        inputs_embeds,
                        attention_mask,
                        spans,
                        fast_state=spectral_state,
                        anchor=spectral_anchor,
                        anchor_available=spectral_anchor_available,
                        unit_ids=spectral_unit_ids,
                    )
                with torch.enable_grad():
                    fast_state = {
                        name: value.detach().clone().requires_grad_(True)
                        for name, value in spectral_state.items()
                    }
                    steps = max(
                        1,
                        int(self.spectral_repair_config.get("update_steps", 1)),
                    )
                    for _ in range(steps):
                        maskpred_loss = spectral_repair.masked_prediction_loss(
                            inputs_embeds,
                            attention_mask,
                            spans,
                            fast_state=fast_state,
                            anchor=spectral_anchor,
                            anchor_available=spectral_anchor_available,
                            unit_ids=spectral_unit_ids,
                            mask_stride=int(
                                self.spectral_repair_config.get("mask_stride", 4)
                            ),
                            reduction="none",
                        )
                        relation_loss, relation_diagnostics = spectral_repair.relation_slack_loss(
                            inputs_embeds,
                            attention_mask,
                            spans,
                            fast_state=fast_state,
                            anchor=spectral_anchor,
                            anchor_available=spectral_anchor_available,
                            reduction="none",
                        )
                        trust_loss = spectral_repair.trust_loss(
                            fast_state,
                            reduction="none",
                        )
                        inner_loss_per_sample = (
                            float(self.spectral_repair_config.get("maskpred_weight", 1.0))
                            * maskpred_loss
                            + float(self.spectral_repair_config.get("slack_weight", 1.0))
                            * relation_loss
                            + float(self.spectral_repair_config.get("trust_weight", 0.01))
                            * trust_loss
                        )
                        # Each row of every fast-state tensor belongs to one
                        # molecule. Summing per-sample objectives preserves the
                        # same row gradient whether that molecule is evaluated
                        # alone or as part of a larger batch.
                        inner_loss = inner_loss_per_sample.sum()
                        gradients = torch.autograd.grad(
                            inner_loss,
                            tuple(fast_state.values()),
                            allow_unused=True,
                        )
                        next_state: dict[str, torch.Tensor] = {}
                        gradient_norms = []
                        for (name, value), gradient in zip(
                            fast_state.items(), gradients
                        ):
                            if gradient is None:
                                gradient = torch.zeros_like(value)
                            gradient_norms.append(gradient.detach().norm(dim=-1))
                            next_state[name] = (
                                value
                                - float(
                                    self.spectral_repair_config.get(
                                        "adaptation_lr", 0.5
                                    )
                                )
                                * gradient
                            ).clamp(-1.0, 1.0).detach().requires_grad_(True)
                        fast_state = next_state
                    spectral_state = {
                        name: value.detach() for name, value in fast_state.items()
                    }
                    with torch.no_grad():
                        (
                            _adapted_repaired,
                            adapted_attention_bias,
                            _adapted_decoder_bias,
                            adapted_repair_diagnostics,
                        ) = spectral_repair.apply(
                            inputs_embeds,
                            attention_mask,
                            spans,
                            fast_state=spectral_state,
                            anchor=spectral_anchor,
                            anchor_available=spectral_anchor_available,
                            unit_ids=spectral_unit_ids,
                        )
                    repair_counts = repair_mask.sum(dim=1).clamp_min(1)
                    gate_change = (
                        (
                            adapted_repair_diagnostics["key_gate"]
                            - baseline_repair_diagnostics["key_gate"]
                        )
                        .abs()
                        .masked_fill(~repair_mask, 0.0)
                        .sum(dim=1)
                        / repair_counts
                    )
                    attention_logit_change = (
                        (adapted_attention_bias - baseline_attention_bias)
                        .abs()
                        .masked_fill(~repair_mask, 0.0)
                        .sum(dim=1)
                        / repair_counts
                    )
                    calibration_ratio_change = (
                        (
                            adapted_repair_diagnostics["calibration_delta_ratio"]
                            - baseline_repair_diagnostics[
                                "calibration_delta_ratio"
                            ]
                        )
                        .abs()
                        .masked_fill(~repair_mask, 0.0)
                        .sum(dim=1)
                        / repair_counts
                    )
                    spectral_diagnostics = {
                        "inner_loss": inner_loss_per_sample.detach(),
                        "maskpred_loss": maskpred_loss.detach(),
                        "relation_loss": relation_loss.detach(),
                        "trust_loss": trust_loss.detach(),
                        "relation_distances": relation_diagnostics.get(
                            "relation_distances"
                        ),
                        "fast_gradient_norm": torch.stack(gradient_norms, dim=1),
                        "fast_state_norm": torch.stack(
                            [
                                value.detach().norm(dim=-1)
                                for value in spectral_state.values()
                            ],
                            dim=1,
                        ),
                        "gate_change": gate_change,
                        "attention_logit_change": attention_logit_change,
                        "calibration_ratio_change": calibration_ratio_change,
                        "steps": steps,
                    }
        tahcd = getattr(self, "tahcd", None)
        if not getattr(self, "_tahcd_runtime_enabled", True):
            tahcd = None
        # Encode once without repair.  The shared encoder is the slow branch;
        # TAHCD's per-sample coefficients are the only fast variables.
        encoder_outputs = self._encode_generation_inputs(
            inputs_embeds,
            attention_mask,
            input_ids,
            apply_tahcd=False,
            spectral_state=spectral_state,
            apply_spectral_repair=spectral_active,
            spectral_anchor=spectral_anchor,
            spectral_anchor_available=spectral_anchor_available,
            spectral_unit_ids=spectral_unit_ids,
        )
        if spectral_diagnostics:
            encoder_outputs["spectral_diagnostics"] = {
                **encoder_outputs.get("spectral_diagnostics", {}),
                **spectral_diagnostics,
            }
        if tahcd is not None:
            tahcd_diagnostics: dict[str, Any] = {}
            tahcd_state = tahcd.initial_fast_state(
                inputs_embeds.shape[0],
                detach=True,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            update_enabled = self._tahcd_inference_enabled
            if update_enabled:
                spans = self._exposure_modality_spans(input_ids)
                pseudo_sequences = None
                if bool(self.tahcd_config.get("ttce_enabled", True)):
                    # TTCE reconstructs a sequence produced by the frozen
                    # source model.  The pseudo sequence is generated before
                    # any fast-state update and never uses target SMILES.
                    with torch.no_grad():
                        pseudo_encoder_outputs = copy.copy(encoder_outputs)
                        pseudo_sequences = self.hf_model.generate(
                            encoder_outputs=pseudo_encoder_outputs,
                            attention_mask=attention_mask,
                            num_beams=1,
                            num_return_sequences=1,
                            generation_config=self.generation_config,
                            use_cache=False,
                        )
                with torch.enable_grad():
                    fast_state = {
                        name: value.detach().clone().requires_grad_(True)
                        for name, value in tahcd_state.items()
                    }
                    steps = max(1, int(self.tahcd_config.get("update_steps", 1)))
                    for _ in range(steps):
                        consistency_loss = tahcd.consistency_loss(
                            encoder_outputs["last_hidden_state"],
                            attention_mask,
                            spans,
                            fast_state=fast_state,
                            prior_weight=float(self.tahcd_config.get("prior_weight", 0.01)),
                        )
                        inner_loss = consistency_loss
                        if pseudo_sequences is not None:
                            adapted_outputs = copy.copy(encoder_outputs)
                            adapted_outputs = self._apply_tahcd_to_encoder_outputs(
                                adapted_outputs,
                                input_ids,
                                attention_mask,
                                fast_state=fast_state,
                            )
                            pseudo_loss = self._tahcd_pseudo_reconstruction_loss(
                                adapted_outputs,
                                attention_mask,
                                pseudo_sequences,
                            )
                            inner_loss = (
                                float(self.tahcd_config.get("ttce_weight", 1.0)) * pseudo_loss
                                + float(self.tahcd_config.get("consistency_weight", 0.25))
                                * consistency_loss
                            )
                        gradients = torch.autograd.grad(
                            inner_loss,
                            tuple(fast_state.values()),
                            allow_unused=True,
                        )
                        gradient_diagnostics = []
                        next_state: dict[str, torch.Tensor] = {}
                        for (name, value), gradient in zip(fast_state.items(), gradients):
                            if gradient is None:
                                gradient_diagnostics.append(torch.zeros_like(value))
                                next_state[name] = value
                            else:
                                gradient_diagnostics.append(gradient.detach())
                                next_state[name] = value - float(
                                    self.tahcd_config.get("adaptation_lr", 0.5)
                                ) * gradient
                            next_state[name] = next_state[name].clamp(
                                tahcd.min_alpha, tahcd.max_alpha
                            ).detach().requires_grad_(True)
                        fast_state = next_state
                    tahcd_state = {name: value.detach() for name, value in fast_state.items()}
                    tahcd_diagnostics["inner_loss"] = inner_loss.detach()
                    tahcd_diagnostics["consistency_loss"] = consistency_loss.detach()
                    tahcd_diagnostics["fast_gradients"] = torch.stack(
                        gradient_diagnostics, dim=1
                    ).squeeze(-1)
                    if pseudo_sequences is not None:
                        tahcd_diagnostics["pseudo_reconstruction_loss"] = pseudo_loss.detach()
            if not update_enabled and not self._tahcd_apply_no_update:
                tahcd_state = None
            encoder_outputs = self._apply_tahcd_to_encoder_outputs(
                encoder_outputs,
                input_ids,
                attention_mask,
                fast_state=tahcd_state,
            )
            block_diagnostics = encoder_outputs.get("tahcd_diagnostics", {})
            if isinstance(block_diagnostics, Mapping):
                tahcd_diagnostics.update(block_diagnostics)
            diagnostics["tahcd"] = tahcd_diagnostics
        return {
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "exposure_diagnostics": diagnostics,
            "tahcd_diagnostics": diagnostics.get("tahcd", {}),
            "spectral_diagnostics": encoder_outputs.get("spectral_diagnostics", {}),
        }

    def score_generated_sequences(
        self,
        batch: Dict[str, Any],
        generated_sequences: torch.Tensor,
        generation_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score fixed pseudo-SMILES prefixes without reading target labels.

        Returns decoder logits and a mask for next-token positions up to EOS.
        This is the generation-task analogue of READ's classification logits.
        """
        if generated_sequences.ndim != 2 or generated_sequences.shape[1] < 2:
            raise ValueError(
                "Generated sequences must have shape [batch, length>=2]."
            )
        if generation_state is None:
            generation_state = self.prepare_generation_state(batch)
        decoder_input = generated_sequences[:, :-1].contiguous()
        next_tokens = generated_sequences[:, 1:].contiguous()
        attention_mask = generation_state["attention_mask"]
        encoder_outputs = generation_state["encoder_outputs"]
        encoder_batch = int(attention_mask.shape[0])
        sequence_batch = int(decoder_input.shape[0])
        if sequence_batch != encoder_batch:
            if sequence_batch % encoder_batch:
                raise ValueError(
                    "Generated sequence count must be a multiple of encoder batch size."
                )
            repeats = sequence_batch // encoder_batch
            attention_mask = attention_mask.repeat_interleave(repeats, dim=0)
            # Candidate-level SGEM scoring evaluates every beam against the same
            # observed spectra. Keep the original state intact because it can be
            # reused by a later differentiable pseudo-sequence pass.
            encoder_outputs = copy.copy(encoder_outputs)
            encoder_outputs["last_hidden_state"] = encoder_outputs[
                "last_hidden_state"
            ].repeat_interleave(repeats, dim=0)
            # CustomModel's decoder also consumes READ's per-token key counts.
            # Every encoder-side batch tensor must follow the beam expansion;
            # leave scalar/metadata fields untouched.
            key_token_counts = encoder_outputs.get("key_token_counts")
            if isinstance(key_token_counts, torch.Tensor):
                encoder_outputs["key_token_counts"] = key_token_counts.repeat_interleave(
                    repeats, dim=0
                )
            key_attention_bias = encoder_outputs.get("key_attention_bias")
            if isinstance(key_attention_bias, torch.Tensor):
                encoder_outputs["key_attention_bias"] = key_attention_bias.repeat_interleave(
                    repeats, dim=0
                )
            state_attention_mask = encoder_outputs.get("attention_mask")
            if (
                isinstance(state_attention_mask, torch.Tensor)
                and state_attention_mask.ndim >= 1
                and state_attention_mask.shape[0] == encoder_batch
            ):
                encoder_outputs["attention_mask"] = state_attention_mask.repeat_interleave(
                    repeats, dim=0
                )
        pad_token_id = self.target_tokenizer.pad_token_id
        decoder_attention_mask = decoder_input.ne(pad_token_id).long()
        token_mask = next_tokens.ne(pad_token_id)
        eos_token_id = self.target_tokenizer.eos_token_id
        if eos_token_id is not None:
            eos = next_tokens.eq(eos_token_id)
            token_mask &= (eos.cumsum(dim=1) - eos.to(dtype=torch.long)) == 0
        model_output = self.hf_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_attention_mask,
            labels=None,
            use_cache=False,
        )
        return model_output.logits, next_tokens, token_mask

    def _token_nll_per_sample(
        self,
        batch: Dict[str, Any],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Mean autoregressive token NLL for each batch item."""
        labels = batch["target"].T.contiguous().to(device=logits.device)
        labels = labels.masked_fill(
            labels.eq(self.target_tokenizer.pad_token_id),
            -100,
        )
        token_loss = functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        valid = labels.ne(-100)
        return token_loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    @staticmethod
    def _qmf_pairwise_rank_loss(
        quality_scores: torch.Tensor,
        sample_nll: torch.Tensor,
        margin: float = 0.1,
    ) -> torch.Tensor:
        """Rank lower-NLL examples above higher-NLL examples by quality."""
        if quality_scores.ndim != 1 or sample_nll.ndim != 1:
            raise ValueError("QMF ranking inputs must be one-dimensional.")
        if quality_scores.shape != sample_nll.shape:
            raise ValueError("QMF quality and NLL inputs must have matching shapes.")
        if quality_scores.numel() < 2:
            return quality_scores.sum() * 0.0
        quality_delta = quality_scores[:, None] - quality_scores[None, :]
        nll_delta = sample_nll[:, None] - sample_nll[None, :]
        pairs = torch.triu(torch.ones_like(nll_delta, dtype=torch.bool), diagonal=1)
        pairs &= nll_delta.ne(0)
        if not pairs.any():
            return quality_scores.sum() * 0.0
        target = -nll_delta.sign()
        losses = functional.relu(float(margin) - target * quality_delta)
        return losses[pairs].mean()

    def _qmf_branch_exclusions(
        self,
        batch: Dict[str, Any],
        canonical_modality: str,
    ) -> frozenset[str]:
        """Keep Formula plus one spectroscopy branch for the QMF auxiliary pass."""
        keep_canonical = {"Formula", canonical_modality}
        excluded = set(self.excluded_input_modalities)
        for data_name in batch["encoder_input"]:
            canonical_name = self.generation_fusion.modality_map.get(
                data_name,
                data_name,
            )
            if canonical_name in {"HNMR", "CNMR"}:
                canonical_name = "NMR"
            if canonical_name not in keep_canonical:
                excluded.add(data_name)
        return frozenset(excluded)

    def _should_run_exposure_meta(self, batch_idx: int) -> bool:
        """Use a deterministic fraction of batches for meta-TTT unrolling."""
        if (
            getattr(self, "exposure_ttt", None) is None
            or float(getattr(self, "_exposure_ttt_meta_batch_fraction", 0.0)) <= 0.0
        ):
            return False
        meta_fraction = float(getattr(self, "_exposure_ttt_meta_batch_fraction", 0.0))
        if meta_fraction >= 1.0:
            return True
        exposure_config = getattr(self, "exposure_ttt_config", {})
        period = int(exposure_config.get("meta_schedule_period", 10))
        if period <= 0:
            raise ValueError("exposure_ttt_config.meta_schedule_period must be positive")
        slots = max(
            1,
            min(period, int(round(period * meta_fraction))),
        )
        return int(batch_idx) % period < slots

    def training_step(self, batch: Dict[str, Any], batch_idx: int = 0) -> torch.Tensor:
        """Training step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            torch.Tensor: loss
        """
        self.train()
        exposure_diagnostics: dict[str, Any] = {}
        if self._should_run_exposure_meta(batch_idx):
            model_output, exposure_diagnostics = self.forward_exposure_ttt(batch)
            loss = exposure_diagnostics.get("outer_loss", model_output.loss)
        else:
            model_output = self.forward(batch)
            loss = model_output.loss

        if loss is None:
            raise RuntimeError("training forward did not produce a loss")

        qmf_enabled = bool(self.qmf_config.get("enabled", False))
        if qmf_enabled:
            if self.generation_fusion is None:
                raise RuntimeError("QMF training requires a generation fusion module.")
            quality_scores = self.generation_fusion.last_quality_scores
            quality_availability = self.generation_fusion.last_quality_availability
            if quality_scores is None or quality_availability is None:
                raise RuntimeError("QMF full forward did not produce quality scores.")

            branch_index = int(self.global_step) % 3
            score_index = branch_index + 1
            canonical_modality = ("NMR", "MSMS", "IR")[branch_index]
            original_exclusions = self.excluded_input_modalities
            try:
                self.excluded_input_modalities = self._qmf_branch_exclusions(
                    batch,
                    canonical_modality,
                )
                branch_output = self.forward(batch)
            finally:
                self.excluded_input_modalities = original_exclusions

            sample_nll = self._token_nll_per_sample(batch, branch_output.logits)
            valid = quality_availability[:, score_index]
            if valid.any():
                auxiliary_loss = sample_nll[valid].mean()
                rank_loss = self._qmf_pairwise_rank_loss(
                    quality_scores[valid, score_index],
                    sample_nll[valid].detach(),
                    margin=float(self.qmf_config.get("rank_margin", 0.1)),
                )
                quality_mean = quality_scores[valid, score_index].mean()
            else:
                auxiliary_loss = sample_nll.sum() * 0.0
                rank_loss = quality_scores[:, score_index].sum() * 0.0
                quality_mean = rank_loss.detach()
            full_loss = loss
            loss = (
                full_loss
                + float(self.qmf_config.get("aux_loss_weight", 1.0)) * auxiliary_loss
                + float(self.qmf_config.get("rank_loss_weight", 0.1)) * rank_loss
            )
            self.log_dict(
                {
                    "train_qmf_full_loss": full_loss,
                    "train_qmf_aux_loss": auxiliary_loss,
                    "train_qmf_rank_loss": rank_loss,
                    "train_qmf_modality_index": loss.new_tensor(branch_index),
                    "train_qmf_quality_mean": quality_mean,
                },
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        self.train_step_outputs = model_output
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            logger=True,
            sync_dist=True,
        )
        if self.generation_fusion is not None:
            self.log(
                "train_generation_fusion_scale",
                self.generation_fusion.output_scale,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )
        if exposure_diagnostics:
            for key, metric_name in (
                ("inner_loss", "train_exposure_inner_loss"),
                ("exposure_loss", "train_exposure_consistency_loss"),
                ("drop_loss", "train_exposure_drop_loss"),
                ("zero_loss", "train_exposure_zero_loss"),
                ("plus_loss", "train_exposure_plus_loss"),
                ("gain_loss", "train_exposure_gain_loss"),
                ("fast_grad_norm", "train_exposure_fast_grad_norm"),
            ):
                value = exposure_diagnostics.get(key)
                if isinstance(value, torch.Tensor):
                    self.log(
                        metric_name,
                        value.float().mean(),
                        prog_bar=key in {"plus_loss", "gain_loss"},
                        on_step=True,
                        logger=True,
                        sync_dist=True,
                    )
            selected = exposure_diagnostics.get("selected_modality")
            if selected is not None:
                print(
                    f"[exposure-ttt] batch={int(batch_idx)} modality={selected} "
                    f"inner={float(exposure_diagnostics['inner_loss']):.6f} "
                    f"zero={float(exposure_diagnostics['zero_loss']):.6f} "
                    f"plus={float(exposure_diagnostics['plus_loss']):.6f} "
                    f"gain={float(exposure_diagnostics['gain_loss']):.6f}",
                    flush=True,
                )
        if isinstance(model_output, CustomLMOutput):
            if model_output.loss_dict:
                for key in model_output.loss_dict.keys():
                    if model_output.loss_dict[key]:
                        self.log(
                            f"train_{key}",
                            model_output.loss_dict[key],
                            prog_bar=True,
                            on_step=True,
                            logger=True,
                            sync_dist=True,
                        )

        return loss
    
    def on_train_epoch_end(self):
        self.log("len_set_sel", len(self.set_sel_indices), prog_bar=True, on_epoch=True, logger=True)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]: # noqa: ARG002
        """Validation step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            Dict[str, Any]: Dictionary containing loss as well as any metrics calculated during validation step
        """

        self.eval()

        exposure_diagnostics: dict[str, Any] = {}
        no_block_output: Optional[Seq2SeqModelOutput] = None
        shuffled_output: Optional[Seq2SeqModelOutput] = None
        validation_controls = bool(
            self.exposure_ttt is not None
            and self._exposure_ttt_validation_compare
            and self._exposure_ttt_validation_controls
        )
        if self.exposure_ttt is not None and self._exposure_ttt_validation_compare:
            # The inner objective is label-free but needs gradients with respect
            # to the temporary fast state.  Validation remains deterministic:
            # exposure masks and shift augmentation both use their deterministic
            # branches.
            with torch.enable_grad():
                model_output, exposure_diagnostics = self.forward_exposure_ttt(
                    batch,
                    requested_modality=self.exposure_ttt_config.get(
                        "validation_modality"
                    ),
                    deterministic=True,
                    no_grad_outer=True,
                )
                if validation_controls:
                    shuffled_output, _ = self.forward_exposure_ttt(
                        batch,
                        requested_modality=self.exposure_ttt_config.get(
                            "validation_modality"
                        ),
                        deterministic=True,
                        shuffle_exposure=True,
                        no_grad_outer=True,
                    )
                    raw_input_ids, raw_attention_mask, raw_embeds = (
                        self._prepare_generation_inputs(
                            batch,
                            apply_modality_dropout=False,
                            apply_exposure_block=False,
                        )
                    )
                    with torch.no_grad():
                        no_block_output = self._forward_prepared(
                            batch,
                            raw_input_ids,
                            raw_attention_mask,
                            raw_embeds,
                        )
        else:
            model_output = self.forward(batch)
        loss = model_output.loss
        if loss is None:
            raise RuntimeError("validation forward did not produce a loss")

        token_correct, token_total = self._calc_token_counts(batch, model_output)
        token_acc = token_correct / token_total.clamp_min(1.0)

        validation_logits_processor = None
        if self.validation_guided_generation and self.guided_generation:
            input_formulas = self._observed_formulas(batch)
            validation_logits_processor = [
                GuidedFormulaProcessor(
                    self.validation_n_beams,
                    input_formulas,
                    self.target_tokenizer,
                )
            ]
        with torch.no_grad():
            generated_sequences = self.generate(
                batch,
                n_beams=self.validation_n_beams,
                logits_processor=validation_logits_processor,
                inner_update_enabled=(
                    True
                    if self.exposure_ttt is not None
                    and self._exposure_ttt_validation_compare
                    else None
                ),
            )

        scores = self.score_val_sequences(
            generated_sequences,
            batch["target"].T,
            n_beams=self.validation_n_beams,
        )

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "val_molecular_accuracy": torch.Tensor([scores["Top-1"]]).to(device=loss.device),
            "val_top5": torch.Tensor(
                [scores.get("Top-5", scores["Top-1"])]
            ).to(device=loss.device),
            "val_top10": torch.Tensor(
                [scores.get("Top-10", scores.get("Top-5", scores["Top-1"]))]
            ).to(device=loss.device),
            "val_topk_score": torch.Tensor(
                [
                    scores["Top-1"]
                    + scores.get("Top-5", scores["Top-1"])
                    + scores.get("Top-10", scores.get("Top-5", scores["Top-1"]))
                ]
            ).to(device=loss.device),
            "_batch_size": len(batch["target_smiles"]),
            "_val_token_correct": token_correct,
            "_val_token_total": token_total,
        }
        if exposure_diagnostics:
            zero_output = exposure_diagnostics.get("zero_output")
            if zero_output is not None:
                zero_loss = zero_output.loss
                zero_correct, zero_total = self._calc_token_counts(batch, zero_output)
                with torch.no_grad():
                    zero_sequences = self.generate(
                        batch,
                        n_beams=self.validation_n_beams,
                        logits_processor=validation_logits_processor,
                        inner_update_enabled=False,
                        exposure_block_enabled=True,
                    )
                    zero_scores = self.score_val_sequences(
                        zero_sequences,
                        batch["target"].T,
                        n_beams=self.validation_n_beams,
                    )
                val_outputs["val_loss_no_update"] = zero_loss.detach()
                val_outputs["val_token_acc_no_update"] = (
                    zero_correct / zero_total.clamp_min(1.0)
                ).detach()
                val_outputs["val_delta_loss_update_minus_no_update"] = (
                    loss.detach() - zero_loss.detach()
                )
                val_outputs["val_block_no_update_loss"] = zero_loss.detach()
                val_outputs["val_block_no_update_token_acc"] = (
                    zero_correct / zero_total.clamp_min(1.0)
                ).detach()
                val_outputs["val_block_no_update_molecular_accuracy"] = loss.new_tensor(
                    zero_scores["Top-1"]
                )
                val_outputs["val_block_no_update_top5"] = loss.new_tensor(
                    zero_scores.get("Top-5", zero_scores["Top-1"])
                )
                val_outputs["val_block_no_update_top10"] = loss.new_tensor(
                    zero_scores.get(
                        "Top-10", zero_scores.get("Top-5", zero_scores["Top-1"])
                    )
                )
                val_outputs["_val_block_no_update_token_correct"] = zero_correct.detach()
                val_outputs["_val_block_no_update_token_total"] = zero_total.detach()

                if validation_controls and no_block_output is not None:
                    no_block_loss = no_block_output.loss
                    if no_block_loss is None:
                        raise RuntimeError("No-block validation forward did not produce a loss")
                    no_block_correct, no_block_total = self._calc_token_counts(
                        batch, no_block_output
                    )
                    with torch.no_grad():
                        no_block_sequences = self.generate(
                            batch,
                            n_beams=self.validation_n_beams,
                            logits_processor=validation_logits_processor,
                            inner_update_enabled=False,
                            exposure_block_enabled=False,
                        )
                        no_block_scores = self.score_val_sequences(
                            no_block_sequences,
                            batch["target"].T,
                            n_beams=self.validation_n_beams,
                        )
                    no_block_token_acc = no_block_correct / no_block_total.clamp_min(1.0)
                    val_outputs.update(
                        {
                            "val_no_block_loss": no_block_loss.detach(),
                            "val_no_block_token_acc": no_block_token_acc.detach(),
                            "val_no_block_molecular_accuracy": loss.new_tensor(
                                no_block_scores["Top-1"]
                            ),
                            "val_no_block_top5": loss.new_tensor(
                                no_block_scores.get("Top-5", no_block_scores["Top-1"])
                            ),
                            "val_no_block_top10": loss.new_tensor(
                                no_block_scores.get(
                                    "Top-10", no_block_scores.get("Top-5", no_block_scores["Top-1"])
                                )
                            ),
                            "_val_no_block_token_correct": no_block_correct.detach(),
                            "_val_no_block_token_total": no_block_total.detach(),
                        }
                    )

                if validation_controls and shuffled_output is not None:
                    shuffled_loss = shuffled_output.loss
                    if shuffled_loss is None:
                        raise RuntimeError("Shuffled validation forward did not produce a loss")
                    shuffled_correct, shuffled_total = self._calc_token_counts(
                        batch, shuffled_output
                    )
                    with torch.no_grad():
                        shuffled_sequences = self.generate(
                            batch,
                            n_beams=self.validation_n_beams,
                            logits_processor=validation_logits_processor,
                            inner_update_enabled=True,
                            exposure_block_enabled=True,
                            shuffle_exposure=True,
                        )
                        shuffled_scores = self.score_val_sequences(
                            shuffled_sequences,
                            batch["target"].T,
                            n_beams=self.validation_n_beams,
                        )
                    shuffled_token_acc = shuffled_correct / shuffled_total.clamp_min(1.0)
                    val_outputs.update(
                        {
                            "val_shuffled_loss": shuffled_loss.detach(),
                            "val_shuffled_token_acc": shuffled_token_acc.detach(),
                            "val_shuffled_molecular_accuracy": loss.new_tensor(
                                shuffled_scores["Top-1"]
                            ),
                            "val_shuffled_top5": loss.new_tensor(
                                shuffled_scores.get("Top-5", shuffled_scores["Top-1"])
                            ),
                            "val_shuffled_top10": loss.new_tensor(
                                shuffled_scores.get(
                                    "Top-10", shuffled_scores.get("Top-5", shuffled_scores["Top-1"])
                                )
                            ),
                            "_val_shuffled_token_correct": shuffled_correct.detach(),
                            "_val_shuffled_token_total": shuffled_total.detach(),
                        }
                    )
        if isinstance(model_output, CustomLMOutput):
            if model_output.loss_dict:
                for key in model_output.loss_dict.keys():
                    val_outputs[f"val_{key}"] = model_output.loss_dict[key]

        self._update_ms_validation_metrics(batch, model_output)

        self.validation_step_outputs.append(val_outputs)
        return val_outputs

    def on_validation_epoch_end(self):
        avg_outputs = self._avg_dicts(self.validation_step_outputs)
        token_correct, token_total = self._aggregate_token_counts(
            self.validation_step_outputs
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                token_correct, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                token_total, op=torch.distributed.ReduceOp.SUM
            )
        avg_outputs["val_token_acc"] = token_correct / token_total.clamp_min(1.0)
        avg_outputs.pop("_val_token_correct", None)
        avg_outputs.pop("_val_token_total", None)
        for prefix in ("block_no_update", "no_block", "shuffled"):
            correct_key = f"_val_{prefix}_token_correct"
            total_key = f"_val_{prefix}_token_total"
            if correct_key not in avg_outputs or total_key not in avg_outputs:
                continue
            control_correct = torch.stack(
                [output[correct_key] for output in self.validation_step_outputs]
            ).sum()
            control_total = torch.stack(
                [output[total_key] for output in self.validation_step_outputs]
            ).sum()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    control_correct, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    control_total, op=torch.distributed.ReduceOp.SUM
                )
            avg_outputs[f"val_{prefix}_token_acc"] = (
                control_correct / control_total.clamp_min(1.0)
            )
            avg_outputs.pop(correct_key, None)
            avg_outputs.pop(total_key, None)
        self._log_dict(avg_outputs)
        self._log_ms_validation_metrics()
        # CSVLogger writes validation metrics to worker-local storage. Keep a
        # compact stdout copy so remote job monitors can observe progress
        # without requiring access to the worker filesystem.
        metric_names = (
            "val_loss",
            "val_loss_no_update",
            "val_delta_loss_update_minus_no_update",
            "val_token_acc",
            "val_token_acc_no_update",
            "val_molecular_accuracy",
            "val_top5",
            "val_top10",
            "val_no_block_loss",
            "val_no_block_token_acc",
            "val_no_block_molecular_accuracy",
            "val_no_block_top5",
            "val_no_block_top10",
            "val_block_no_update_loss",
            "val_block_no_update_token_acc",
            "val_block_no_update_molecular_accuracy",
            "val_block_no_update_top5",
            "val_block_no_update_top10",
            "val_shuffled_loss",
            "val_shuffled_token_acc",
            "val_shuffled_molecular_accuracy",
            "val_shuffled_top5",
            "val_shuffled_top10",
        )
        metric_text = []
        for name in metric_names:
            value = avg_outputs.get(name)
            if torch.is_tensor(value):
                value = value.detach().float().mean().item()
            if value is not None:
                metric_text.append(f"{name}={float(value):.6f}")
        if metric_text:
            print(
                f"[validation] epoch={self.current_epoch} "
                + " ".join(metric_text),
                flush=True,
            )
        self.validation_step_outputs = list()

    def _update_ms_validation_metrics(
        self,
        batch: Dict[str, Any],
        model_output: Seq2SeqModelOutput,
    ) -> None:
        if self.ms_validation_metrics is None:
            return
        retrieval_predictions = model_output.get("retrieval_predictions")
        if not isinstance(retrieval_predictions, Mapping):
            return
        task_logits = retrieval_predictions.get("task_logits")
        targets = batch.get("retrieval_targets")
        if not isinstance(task_logits, Mapping) or not isinstance(targets, Mapping):
            return
        logits = task_logits.get("MSMS")
        target = targets.get("MSMS")
        if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
            return

        availability = retrieval_predictions.get("modality_availability")
        if isinstance(availability, Mapping):
            mask = availability.get("MSMS")
            if isinstance(mask, torch.Tensor):
                mask = mask.to(device=logits.device, dtype=torch.bool)
                logits = logits[mask]
                target = target[mask]
        if logits.shape[0] > 0:
            self.ms_validation_metrics.update(targets=target, logits=logits)
            self._ms_validation_sample_count += int(logits.shape[0])

    def _log_ms_validation_metrics(self) -> None:
        if self.ms_validation_metrics is None:
            return
        if self._ms_validation_sample_count == 0:
            self.ms_validation_metrics.reset()
            return
        metrics = self.ms_validation_metrics.compute()
        for name in ("macro_auprc", "macro_f1", "macro_balanced_accuracy"):
            self.log(
                f"val_ms_{name}",
                metrics[name],
                logger=True,
                sync_dist=False,
            )
        self.ms_validation_metrics.reset()
        self._ms_validation_sample_count = 0

    def predict_step(self, batch, batch_idx): # noqa: ARG002
        """Predict step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            Dict[str, Any]: Dictionary containing loss as well as any metrics calculated during test step
        """

        self.eval()

        model_output = self.forward(batch)
        loss = model_output.loss
        
        if self.guided_generation:
            input_formulas = self._observed_formulas(batch)
            logit_processor = [
                GuidedFormulaProcessor(
                    self.n_beams,
                    input_formulas,
                    self.target_tokenizer,
                )
            ]
            generated_sequences = self.generate(batch, n_beams=self.n_beams, logits_processor=logit_processor)
        else:
            generated_sequences = self.generate(batch, n_beams=self.n_beams)
        
        decoded_sequences = self.target_tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

        extra = {}
        for key in batch.keys():
            if not key.startswith("encoder_") and not key.startswith("decoder_") and not key.startswith("target_"):
                extra[key] = batch[key]
                
        
        return {"loss": loss, "predictions": decoded_sequences, "targets": batch['target_smiles'], **extra}

    def _avg_dicts(self, colls: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = set().union(*(coll.keys() for coll in colls))
        complete_dict: Dict[str, list] = {
            key: [] for key in keys if key != "_batch_size"
        }
        weights = [float(coll.get("_batch_size", 1)) for coll in colls]
        for coll in colls:
            for key in complete_dict.keys():
                complete_dict[key].append(coll.get(key))

        total_weight = sum(weights)
        avg_dict = {
            key: sum(value * weight for value, weight in zip(metric, weights))
            / total_weight
            for key, metric in complete_dict.items()
            if not any(value is None for value in metric)
        }
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            if key == "val_molecular_accuracy":
                self.log(
                    "val_molecular_accuracy",
                    val,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            elif val is not None:
                self.log(key, val, sync_dist=True)

    def score_val_sequences(
        self,
        generated_sequences: torch.Tensor,
        targets: Union[List[str], torch.Tensor],
        n_beams: int,
    ) -> Dict[str, float]:
        # Move to evaluator
        """Decodes generated sequences and calculates TopN scores.
        Args:
            generated_sequences: sampled sequences from the model
            target: target sequences
            n_beams: n beams used in generation
        Returns:
            Dict[str, float]: Dictionary containing the TopN scores
        """

        # Decode Targets
        # Do not mutate the collated validation batch while decoding labels.
        if torch.is_tensor(targets):
            targets = targets.clone()
            targets[targets == -100] = self.target_tokenizer.pad_token_id
            decoded_targets = self.target_tokenizer.batch_decode(
                targets, skip_special_tokens=True
            )
        else:
            decoded_targets = [str(target) for target in targets]

        # Decode Predictions
        decoded_sequences = self.target_tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        # Reshape to (batch_size, n_beams)
        if n_beams < 1:
            raise ValueError("n_beams must be at least 1")
        expected_predictions = len(decoded_targets) * n_beams
        if len(decoded_sequences) != expected_predictions:
            raise RuntimeError(
                "Generation returned an unexpected number of sequences: "
                f"expected={expected_predictions}, got={len(decoded_sequences)}"
            )
        decoded_sequences = [
            decoded_sequences[i * n_beams : (i + 1) * n_beams]
            for i in range(len(decoded_targets))
        ]
        
        # Molecular accuracy must compare RDKit-canonical molecules.  String
        # exact-match is not invariant to valid SMILES serialization choices.
        scores = calc_sampling_metrics(
            decoded_sequences, decoded_targets, molecules=True
        )

        return scores
    
    def _calc_token_counts(self, batch_input, model_output):
        token_ids = batch_input["target"].T
        pred_tokens = torch.argmax(model_output.logits, dim=-1)

        target_mask = token_ids != -100
        if self.target_tokenizer.pad_token_id is not None:
            target_mask &= token_ids != self.target_tokenizer.pad_token_id
        correct_ids = torch.eq(token_ids, pred_tokens) & target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        return num_correct, total

    def _calc_token_acc(self, batch_input, model_output):
        num_correct, total = self._calc_token_counts(batch_input, model_output)
        return num_correct / total.clamp_min(1.0)

    @staticmethod
    def _aggregate_token_counts(validation_outputs):
        num_correct = torch.stack(
            [output["_val_token_correct"] for output in validation_outputs]
        ).sum()
        total = torch.stack(
            [output["_val_token_total"] for output in validation_outputs]
        ).sum()
        return num_correct, total
