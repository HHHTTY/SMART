import json
import random
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import faiss  # type:ignore
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
try:
    from activeft.sift import Retriever  # type:ignore
except ModuleNotFoundError:  # optional dependency; unused by ordinary training
    class Retriever:  # type:ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("activeft is required only for active-learning selection")
from datasets import Dataset, DatasetDict, IterableDataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..configuration import DEFAULT_SETTINGS
from ..modeling.multitask_retrieval import (
    MultimodalCandidateRetriever,
    encode_formula_compositions,
)
from .data_utils import IterableDatasetWithLength
from .preprocessors import PatchPreprocessor


def _move_nested(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    return value


def _to_batch_first(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.transpose(1, 0) if value.ndim >= 2 else value
    if isinstance(value, dict):
        return {key: _to_batch_first(item) for key, item in value.items()}
    return value


@dataclass
class MultiModalDataCollator:
    preprocessors: Dict[str, Any]
    data_config: Dict[str, Any]
    model_type: str

    dataset: InitVar[DatasetDict]
    extra_columns: Optional[List[str]] = None


    padding: bool = True
    max_source_length: Optional[Dict[str, int]] = None
    max_target_length: Optional[int] = None
    return_tensors: str = "pt"

    input_modalities: List[str] = field(init=False)
    target_modality: str = field(init=False)
    alignment_modality: List[str] = field(init=False)
    retrieval_target_modalities: Dict[str, str] = field(init=False)

    def __post_init__(self, dataset: DatasetDict):
        """
        Determines the input and target modalities. If not provided computes the max_source and max_target length.
        """
        input_modalities = [
            modality
            for modality, modality_config in self.data_config.items()
            if not modality_config["target"] and "retrieval_target" not in modality_config
        ]
        target_modality_list = [
            modality
            for modality, modality_config in self.data_config.items()
            if modality_config["target"] and ("alignment" not in modality_config or not modality_config["alignment"])
        ]
        alignment_modality_list = [
            modality
            for modality, modality_config in self.data_config.items()
            if modality_config["target"] and "alignment" in modality_config and modality_config["alignment"]
        ]
        retrieval_target_modalities = {
            modality: modality_config["retrieval_target"]
            for modality, modality_config in self.data_config.items()
            if "retrieval_target" in modality_config
        }
        if len(alignment_modality_list) > 1:
            raise ValueError("At most 1 target alignment modality can be specified.")
        if len(target_modality_list) != 1:
            raise ValueError("Only 1 target modality can be specified.")

        target_modality = target_modality_list[0]

        self.input_modalities = input_modalities
        self.target_modality = target_modality
        self.alignment_modality = alignment_modality_list
        self.retrieval_target_modalities = retrieval_target_modalities

        # Compute max source length if not provided
        if self.max_source_length is None:
            self.max_source_length = self.compute_source_lengths(
                dataset[list(dataset.keys())[0]]
            )

        # Compute max target length if not provided; Only relevant for Text as output
        if (
            self.max_target_length is None
            and self.data_config[self.target_modality]["type"] == "text"
        ):
            self.max_target_length = self.compute_target_length(
                dataset[list(dataset.keys())[0]]
            )

    def compute_source_lengths(self, dataset: Dataset) -> Dict[str, int]:
        max_lengths = dict()

        if isinstance(dataset, IterableDatasetWithLength):
            num_samples = min(DEFAULT_SETTINGS.default_samples, dataset._length)
            sampled_dataset = dataset.take(num_samples)
            sampled_dataset = Dataset.from_generator(lambda: sampled_dataset.__iter__(), split=dataset.split)
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        # Compute the max length of each modality
        for modality in self.input_modalities:
            if self.data_config[modality]["type"] == "text":
                for sample in sampled_dataset[modality]:
                    tokenized_sample = self.preprocessors[modality](
                        text=sample, padding=False
                    )["input_ids"]
                    if modality not in max_lengths:
                        max_lengths[modality] = len(tokenized_sample) + 5
                    else:
                        if (len(tokenized_sample) + 5) > max_lengths[modality]:
                            max_lengths[modality] = len(tokenized_sample) + 5
            elif self.data_config[modality]["type"] == "1D_patches":
                sample = sampled_dataset.select([0])[modality]
                processed_sample, _ = self.preprocessors[modality](sample)
                max_length_patches = processed_sample.shape[1]
                max_lengths[modality] = max_length_patches

        return max_lengths

    def compute_target_length(self, dataset: Dataset) -> int:
        # Determine the max length of the target modality

        max_target_length = 0

        if isinstance(dataset, IterableDatasetWithLength):
            num_samples = min(DEFAULT_SETTINGS.default_samples, dataset._length)
            sampled_dataset = dataset.take(num_samples)
            sampled_dataset = Dataset.from_generator(lambda: sampled_dataset.__iter__(), split=dataset.split)
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        for sample in sampled_dataset[self.target_modality]:
            tokenized_sample = self.preprocessors[self.target_modality](
                text=sample, padding=False
            )["input_ids"]
            if len(tokenized_sample) > max_target_length:
                max_target_length = len(tokenized_sample)

        return max_target_length + 5
    
    def __call__(
        self, batch: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch_dict = {
            k: [batch[i][k] for i in range(len(batch))] for k, v in batch[0].items()
        }


        # Prepare Encoder and target
        input_dict, global_input_attention_mask, modality_attention_masks = self.prepare_encoder_input(
            batch_dict, return_tensors
        )
        modality_availability = {
            modality: (~pad_mask).any(dim=0)
            for modality, pad_mask in modality_attention_masks.items()
        }

        retrieval_targets = {
            task_name: torch.as_tensor(
                np.asarray(batch_dict[modality]), dtype=torch.float32
            )
            for modality, task_name in self.retrieval_target_modalities.items()
        }


        alignment_input = None
        if len(self.alignment_modality) == 1:
            alignment_input = torch.tensor(np.array(batch_dict[self.alignment_modality[0]]))
            if isinstance(self.preprocessors[self.alignment_modality[0]], PatchPreprocessor) and alignment_input.shape[1] < 1800:
                alignment_input = torch.nn.functional.pad(alignment_input, (0, 1800 - alignment_input.shape[1]), "constant", 0)

            if self.data_config[self.alignment_modality[0]]["type"] == "1D_patches" and self.preprocessors[self.alignment_modality[0]].interplation_merck:
                alignment_input = torch.tensor(self.preprocessors[self.alignment_modality[0]].interpolation_merck(alignment_input), dtype=torch.float32)
        target_tensor = self.prepare_target(batch_dict, return_tensors)

        # Prepare batches for BART or encoder only model
        if self.model_type in [
            "BART",
            "BartForConditionalGeneration",
            "CustomBartForConditionalGeneration",
            "T5ForConditionalGeneration",
            "CustomModel"
        ]:
            tokenized_label_input_ids = target_tensor["input_ids"].transpose(0, 1)

            # Construct decoder input as dict to conform with model wrapper embedding logic
            decoder_input = {self.target_modality: tokenized_label_input_ids[:-1, :]}

            decoder_pad_mask = (
                ~target_tensor["attention_mask"].transpose(0, 1).type(torch.bool)
            )

            if self.data_config[self.target_modality]["type"] == "carbon":
                target = self.preprocessors[self.target_modality].process_carbon(
                    batch_dict[self.target_modality]
                )
            elif self.data_config[self.target_modality]["type"] == "multiplets":
                target = self.preprocessors[self.target_modality].process_multiplets(
                    batch_dict[self.target_modality],
                    encoding=self.preprocessors[self.target_modality].encoding,
                    j_values=self.preprocessors[self.target_modality].j_values,
                )[0]
            else:
                target = batch_dict[self.target_modality]

            return_dict =  {
                "encoder_input": input_dict,
                "encoder_pad_mask": global_input_attention_mask,
                "encoder_modality_pad_masks": modality_attention_masks,
                "encoder_modality_availability": modality_availability,
                "decoder_input": decoder_input,
                "decoder_pad_mask": decoder_pad_mask[:-1, :],
                "target": tokenized_label_input_ids.clone()[1:, :],
                "target_mask": decoder_pad_mask.clone()[1:, :],
                "target_smiles": target,
            }

            # Keep observable Formula inputs available to generation-time
            # constraints.  This is deliberately sourced from the input
            # modality, never reconstructed from the target SMILES.
            formula_modality = next(
                (
                    modality
                    for modality in self.input_modalities
                    if str(modality).lower() == "formula"
                ),
                None,
            )
            if formula_modality is not None:
                return_dict["input_formulas"] = list(batch_dict[formula_modality])
            
            if alignment_input is not None:
                return_dict["encoder_alignment_input"] = alignment_input
            if retrieval_targets:
                return_dict["retrieval_targets"] = retrieval_targets

            if self.extra_columns and self.extra_columns != [None]:
                for col in self.extra_columns:
                    if col not in return_dict:
                        return_dict[col] = batch_dict[col]
            return return_dict

        elif self.model_type == "encoder":
            return {
                "encoder_input": input_dict,
                "encoder_pad_mask": global_input_attention_mask,
                "encoder_modality_pad_masks": modality_attention_masks,
                "encoder_modality_availability": modality_availability,
                "target": target_tensor,
            }

        else:
            raise ValueError(f"Unknown model type {self.model_type}")

    def prepare_encoder_input(
        self, batch_dict: Dict[str, Any], return_tensors: str
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        input_dict = dict()
        modality_attention_masks = dict()

        # Irregular attention mask
        global_input_attention_mask = None
        for modality in self.input_modalities:
            if self.data_config[modality]["type"] == "text":
                tokenized_modality = self.preprocessors[modality](
                    batch_dict[modality],
                    padding="max_length",
                    max_length=self.max_source_length[modality],  # type: ignore
                    truncation=True,
                    return_tensors=return_tensors,
                )

                tokenized_input = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] in [
                "multiplets",
                "carbon",
                "msms_text",
                "msms_number",
            ]:
                tokenized_modality = self.preprocessors[modality](batch_dict[modality])

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality[
                            "numerical_values"
                        ].transpose(0, 1),
                    }
                else:
                    tokenized_input = tokenized_input_ids

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] in "text_spectrum":
                # Text spectrum requires formula and spectra column as keys in the config

                spectra = batch_dict[self.data_config[modality]["spectra_column"]]

                if self.data_config[modality]["spectra_only"]:
                    formulae = None
                else:
                    formulae = batch_dict[self.data_config[modality]["formula_column"]]

                tokenized_modality = self.preprocessors[modality](
                    formulae=formulae, spectra=spectra
                )

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality[
                            "numerical_values"
                        ].transpose(0, 1),
                    }
                else:
                    tokenized_input = tokenized_input_ids

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] == "peak_positional_encoding":
                spectra = batch_dict[modality]
                tokenized_spectra = self.preprocessors[modality](spectra=spectra)

                tokenized_input = tokenized_spectra["input_ids"].transpose(0, 1)
                token_indices = tokenized_spectra["indices"]
                input_dict[modality] = {
                    "tokenized_input": tokenized_input,
                    "token_indices": token_indices,
                }

                attention_mask = (
                    ~tokenized_spectra["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

            elif self.data_config[modality]["type"] == "run_length_encoding":
                # Text spectrum requires formula and spectra column as keys in the config
                spectra = batch_dict[modality]

                tokenized_modality = self.preprocessors[modality](spectra=spectra)

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                input_dict[modality] = tokenized_input_ids

            elif self.data_config[modality]["type"] == "1D_patches":
                processed_input, attention_mask = self.preprocessors[modality](
                    batch_dict[modality]
                )
                processed_input = processed_input.transpose(0, 1)

                # All spectra have the same length => No masking; Chemformer uses False to indicate when a token is not masked
                attention_mask = attention_mask.transpose(0, 1)

                input_dict[modality] = processed_input

            if global_input_attention_mask is None:
                global_input_attention_mask = attention_mask
            else:
                global_input_attention_mask = torch.cat(
                    (global_input_attention_mask, attention_mask)
                )
            modality_attention_masks[modality] = attention_mask

        return input_dict, global_input_attention_mask, modality_attention_masks

    def prepare_target(self, batch_dict: Dict[str, Any], return_tensors: str) -> Any:
        if self.data_config[self.target_modality]["type"] == "text":
            target_tensor = self.preprocessors[self.target_modality](
                text=batch_dict[self.target_modality],
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
            )
        elif self.data_config[self.target_modality]["type"] in ["carbon", "multiplets"]:
            target_tensor = self.preprocessors[self.target_modality](
                batch_dict[self.target_modality]
            )

        elif self.data_config[self.target_modality]["type"] in [
            "functional_group",
            "class_one_hot",
        ]:
            target = self.preprocessors[self.target_modality](
                batch_dict[self.target_modality]
            )
            target_tensor = torch.Tensor(target)
        elif self.data_config[self.target_modality]["type"] == "no_action":
            target_tensor = torch.Tensor(batch_dict[self.target_modality])

        elif self.data_config[self.target_modality]["type"] == "normalise":
            processed_data = self.preprocessors[self.target_modality](
                np.array(batch_dict[self.target_modality])
            )
            target_tensor = torch.Tensor(processed_data)

        else:
            raise ValueError(f"Unknown Target type: {self.target_modality}")

        return target_tensor


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetDict,
        preprocessors: Dict[str, Union[AutoTokenizer, PatchPreprocessor]],
        data_config: Dict[str, Union[str, bool, int]],
        model_type: str,
        batch_size: int = 128,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        extra_columns: Optional[List[str]] = None,
        num_workers: int = 8,
        validation_batch_size: Optional[int] = None,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        self.dataset = dataset
        self.preprocessors = preprocessors
        self.data_config = data_config
        self.model_type = model_type
        self.batch_size = batch_size
        self.validation_batch_size = (
            batch_size if validation_batch_size is None else int(validation_batch_size)
        )
        if self.validation_batch_size <= 0:
            raise ValueError("validation_batch_size must be a positive integer")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer")
        self.extra_columns = extra_columns

        self.collator = self.get_multimodal_data_collator()

    # Abstract functions that we dont use
    def setup(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    def _worker_options(self) -> Dict[str, Union[bool, int]]:
        if self.num_workers <= 0:
            return {}
        return {
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    def train_dataloader(self) -> DataLoader:

        train_loader = DataLoader(
            self.dataset["train"],
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle = False, #True if not isinstance(self.dataset["train"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            **self._worker_options(),
        )
        return train_loader

    def val_dataloader(
        self,
    ) -> DataLoader:
        
        val_loader = DataLoader(
            self.dataset["validation"],
            collate_fn=self.collator,
            batch_size=self.validation_batch_size,
            shuffle=False if not isinstance(self.dataset["validation"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            **self._worker_options(),
        )
        return val_loader

    def predict_dataloader(
        self,
        test_idx: Optional[Path] = None,
    ) -> DataLoader:

        if "eval_sample" in self.data_config and self.data_config["eval_sample"]:
            if test_idx is None:
                #Sample random 10k samples
                selected_sample = np.random.choice(
                    len(self.dataset["test"]),
                    min(10000, len(self.dataset["test"])),
                    replace=False
                )
            else:
                with test_idx.open("rb") as f:
                    selected_sample = np.load(f)
            selected_test_set = self.dataset["test"].select(selected_sample)
        else:
            selected_test_set = self.dataset["test"]

        test_loader = DataLoader(
            selected_test_set,
            collate_fn=self.collator,
            # Prediction callers choose the batch size; the old hard cap of 64
            # left large-memory GPUs mostly idle during beam search.
            batch_size=self.batch_size,
            shuffle=False, # if not isinstance(selected_test_set, (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            **self._worker_options(),
        )
        return test_loader

    def get_multimodal_data_collator(self) -> MultiModalDataCollator:
        data_collator = MultiModalDataCollator(
            preprocessors=self.preprocessors,
            data_config=self.data_config,
            dataset=self.dataset,
            model_type=self.model_type,
            extra_columns=self.extra_columns
        )
        return data_collator

class TTTMultiModalDataModule(MultiModalDataModule):
    """Class to perform test-time tuning on the given dataset.
    """
    def __init__(
        self,
        model,
        dataset: DatasetDict,
        preprocessors: Dict[str, Union[AutoTokenizer, PatchPreprocessor]],
        data_config: Dict[str, Union[str, bool, int]],
        model_type: str,
        batch_size: int = 128,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        extra_columns: Optional[List[str]] = None,
        num_workers: int = 8,
        device: str = 'cpu',
        reduced_val: bool = False,
        only_faiss: bool = True,
        path_selection: str = None,
        similarity_criterion: str = 'embeddings',
        nearest_neighbors: int = None,
        retrieval_config: Optional[Mapping[str, Any]] = None,
        validation_batch_size: Optional[int] = None,
    ):
        # inherit all the methods from MultiModalDataModule
        super().__init__(
            dataset,
            preprocessors,
            data_config,
            model_type,
            batch_size,
            max_source_length,
            max_target_length,
            extra_columns,
            num_workers,
            validation_batch_size,
        )

        self.model = model
        self.device = device
        self.num_workers = num_workers
        self.only_faiss = only_faiss
        self.path_selection = path_selection
        self.similarity_criterion = similarity_criterion
        self.nearest_neighbors = nearest_neighbors
        self.retrieval_config = dict(retrieval_config or {})

        if reduced_val:
            if len(self.dataset["validation"]) > self.validation_batch_size:
                indices_val = random.sample(
                    range(len(self.dataset["validation"])), self.validation_batch_size
                )  # using 1 batch
                self.dataset["validation"] = self.dataset["validation"].select(indices_val)
                logger.info(
                    f"Using only {self.validation_batch_size} samples from validation set"
                )

        # implement anyways the standard datamodule, needed to access the processed data
        self.datamodule = MultiModalDataModule(
            dataset=self.dataset,
            preprocessors=self.preprocessors,
            data_config=self.data_config,
            model_type=self.model_type,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            extra_columns=self.extra_columns,
            validation_batch_size=self.validation_batch_size,
        )
        
        self.epochs = 0
        self.d = self.model.hf_model.encoder.norm.normalized_shape[0] # Dimension of the vectors
        if self.similarity_criterion == "fingerprints":
            if self.model.hf_model.align_network is None:
                raise ValueError("Fingerprint TTT requires an alignment fingerprint head.")
            self.d_fp = self.model.hf_model.align_network[2].out_features
        elif self.similarity_criterion == "multitask_retrieval":
            if self.model.multitask_retrieval_heads is None:
                raise ValueError(
                    "Multitask retrieval TTT requires multitask_retrieval_config."
                )
            self.d_fp = self.model.multitask_retrieval_heads.fingerprint_dim
        else:
            self.d_fp = self.d


    def make_embeddings(self, dim, batches, device, save=None):
        """Make embeddings for vectors in the test set for compression mode mean
        """
        embeddings = torch.empty((0, dim)).to(device)
        for batch in iter(batches):
            embeddings_chunk, att_mask = self.make_embeddings_datamodule(batch, save)
            embeddings_chunk = torch.stack([torch.mean(vec[att==1], dim=0) for vec, att in zip(embeddings_chunk, att_mask)])
            embeddings = torch.cat((embeddings, embeddings_chunk), 0)
        
        return embeddings.detach().cpu()

    def make_embeddings_datamodule(self, batch, save=None):
        """Make the embeddings of the data given the model, using the dataloader (takes care of the batches automatically)
        """
        with torch.no_grad():
            model_batch = _move_nested(batch, self.device)
            input_ids, attention_mask = self.model.prepare_encoder_inputs(model_batch)
            input_embeds = self.model.multimodal_embedding(input_ids)
            embeddings = self.model.hf_model.encode(input_embeds, attention_mask)
        
        if save and isinstance(save, str):
            data = pd.DataFrame()
            data['last_hidden_state'] = embeddings.last_hidden_state.cpu().numpy().tolist()
            data['attention_mask'] = embeddings.attention_mask.cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/{save}_embeddings_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(data.to_json(), f)
            f.close()

        return embeddings.last_hidden_state, embeddings.attention_mask

    def get_fingerprints(self, loader):

        embeddings = self.make_embeddings(self.d, loader, self.device)
        fingerprints = self.model.hf_model.predict_fingerprint(embeddings=embeddings)
        
        del embeddings

        return fingerprints

    def get_multitask_predictions(self, loader) -> Dict[str, Any]:
        """Predict retrieval signals from spectra without reading test labels."""
        fused_logits = []
        modality_fingerprint_logits: Dict[str, List[torch.Tensor]] = {}
        modality_weights = []
        retrieval_availability = []
        task_logits: Dict[str, List[torch.Tensor]] = {}
        modality_availability: Dict[str, List[torch.Tensor]] = {}
        modality_order = None
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                model_batch = _move_nested(batch, self.device)
                predictions = self.model.predict_multitask_retrieval(model_batch)
                fused_logits.append(
                    predictions["fused_fingerprint_logits"].detach().cpu()
                )
                modality_weights.append(predictions["modality_weights"].detach().cpu())
                batch_availability = predictions.get("modality_availability", {})
                if not isinstance(batch_availability, dict):
                    raise TypeError("modality_availability must be a dictionary.")
                batch_retrieval_available = torch.stack(
                    [
                        values.detach().to(dtype=torch.bool)
                        for values in batch_availability.values()
                    ],
                    dim=1,
                ).any(dim=1)
                batch_retrieval_available &= torch.isfinite(
                    predictions["fused_fingerprint_logits"]
                ).all(dim=1)
                retrieval_availability.append(batch_retrieval_available.cpu())
                modality_order = predictions["modality_order"]
                for name, values in predictions["modality_fingerprint_logits"].items():
                    modality_fingerprint_logits.setdefault(name, []).append(
                        values.detach().cpu()
                    )
                for name, values in predictions["task_logits"].items():
                    task_logits.setdefault(name, []).append(values.detach().cpu())
                for name, values in batch_availability.items():
                    modality_availability.setdefault(name, []).append(
                        values.detach().to(dtype=torch.bool).cpu()
                    )
        if was_training:
            self.model.train()
        if not fused_logits or modality_order is None:
            raise ValueError("The retrieval prediction loader produced no batches.")
        return {
            "fused_fingerprint_logits": torch.cat(fused_logits, dim=0),
            "modality_fingerprint_logits": {
                name: torch.cat(values, dim=0)
                for name, values in modality_fingerprint_logits.items()
            },
            "task_logits": {
                name: torch.cat(values, dim=0) for name, values in task_logits.items()
            },
            "modality_weights": torch.cat(modality_weights, dim=0),
            "retrieval_availability": torch.cat(retrieval_availability, dim=0),
            "modality_order": modality_order,
            "modality_availability": {
                name: torch.cat(values, dim=0)
                for name, values in modality_availability.items()
            },
        }

    def _retrieval_fingerprint_probabilities(
        self, predictions: Mapping[str, Any]
    ) -> torch.Tensor:
        """Convert retrieval-head outputs to the configured probability representation."""
        representation = str(
            self.retrieval_config.get("fingerprint_representation", "fused")
        )
        if representation == "fused":
            return torch.sigmoid(predictions["fused_fingerprint_logits"])
        if representation != "modality_blend":
            raise ValueError(
                "fingerprint_representation must be 'fused' or 'modality_blend'."
            )

        modality_logits = predictions.get("modality_fingerprint_logits", {})
        modality_availability = predictions.get("modality_availability", {})
        modality_weights = dict(
            self.retrieval_config.get(
                "fingerprint_modality_weights",
                {"MSMS": 1.0, "NMR": 1.0, "IR": 1.0},
            )
        )
        numerator = None
        denominator = None
        for name in predictions["modality_order"]:
            if name not in modality_logits:
                continue
            probability = torch.sigmoid(modality_logits[name])
            available = modality_availability.get(
                name,
                torch.ones(probability.shape[0], dtype=torch.bool),
            ).to(dtype=probability.dtype).unsqueeze(1)
            weight = float(modality_weights.get(name, 0.0))
            weighted = probability * available * weight
            numerator = weighted if numerator is None else numerator + weighted
            contribution = available * weight
            denominator = (
                contribution
                if denominator is None
                else denominator + contribution
            )
        if numerator is None or denominator is None:
            raise ValueError("No modality fingerprint is available for blending.")
        return numerator / denominator.clamp_min(1e-8)

    def _split_prediction_loader(self, split: str) -> DataLoader:
        """Build a deterministic loader whose output rows match the split order."""
        return DataLoader(
            self.dataset[split],
            collate_fn=self.collator,
            batch_size=min(self.batch_size, 64),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def get_predicted_candidate_retrieval_bank(
        self, split: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """Predict source representations in the same space as retrieval queries."""
        predictions = self.get_multitask_predictions(
            self._split_prediction_loader(split)
        )
        fingerprints = self._retrieval_fingerprint_probabilities(predictions)
        retrieval_available = predictions["retrieval_availability"].to(dtype=torch.bool)
        fingerprints = fingerprints.masked_fill(
            ~retrieval_available.unsqueeze(1), float("nan")
        )
        tasks: Dict[str, torch.Tensor] = {}
        for name, logits in predictions["task_logits"].items():
            values = torch.sigmoid(logits)
            available = predictions.get("modality_availability", {}).get(name)
            if available is not None:
                values = values.masked_fill(
                    ~available.to(dtype=torch.bool).unsqueeze(1), float("nan")
                )
            tasks[name] = values

        expected = len(self.dataset[split])
        if fingerprints.shape[0] != expected or any(
            values.shape[0] != expected for values in tasks.values()
        ):
            raise RuntimeError(
                f"Predicted retrieval bank row count does not match {split!r} split."
            )
        return fingerprints, tasks, predictions

    def _refresh_multitask_predictions(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refresh symmetric source/query predictions from the current model."""
        (
            self.candidate_fingerprints,
            self.candidate_tasks,
            self.candidate_retrieval_predictions,
        ) = self.get_predicted_candidate_retrieval_bank("train")
        self.test_retrieval_predictions = self.get_multitask_predictions(
            self._split_prediction_loader("test")
        )
        fingerprints_test = self._retrieval_fingerprint_probabilities(
            self.test_retrieval_predictions
        )
        self.train_tensor = torch.nn.functional.normalize(
            torch.nan_to_num(self.candidate_fingerprints), p=2, dim=1
        )
        self.test_tensor = torch.nn.functional.normalize(
            torch.nan_to_num(fingerprints_test), p=2, dim=1
        )
        return self.candidate_fingerprints, fingerprints_test

    def get_candidate_retrieval_bank(
        self, split: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Read structure-derived labels only from the labeled source split."""
        targets: Dict[str, torch.Tensor] = {}
        for modality, task_name in self.collator.retrieval_target_modalities.items():
            values = np.asarray(self.dataset[split][modality], dtype=np.float32)
            targets[task_name] = torch.as_tensor(values, dtype=torch.float32)
        if "fingerprint" not in targets:
            raise KeyError("A fingerprint retrieval target is required for source candidates.")

        fingerprint = targets.pop("fingerprint")
        return fingerprint, targets

class KmeansTTTMultiModalDataModule(TTTMultiModalDataModule):
    """Class to perform K-means clustering on the test set of the given dataset,
    to then use only one point per cluster to perform test-time tuning.
    """
    def __init__(
            self,
            model,
            dataset,
            preprocessors,
            data_config,
            model_type,
            batch_size = 128,
            max_source_length = None,
            max_target_length = None,
            extra_columns = None,
            num_workers = 8,
            device = 'cpu',
            reduced_val = False,
            only_faiss = True,
            path_selection = None,
            similarity_criterion = 'fingerprints',
            nearest_neighbors = None,
            n_clusters = None,
            n_test_points = None,
            n_train_points = None,
            update_embeds = 10, # False or int
            seed = 3247,
            retrieval_config: Optional[Mapping[str, Any]] = None,
            validation_batch_size: Optional[int] = None,
        ):
        
        super().__init__(
            model=model,
            dataset=dataset,
            preprocessors=preprocessors,
            data_config=data_config,
            model_type=model_type,
            batch_size=batch_size,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            extra_columns=extra_columns,
            num_workers=num_workers,
            device=device,
            reduced_val=reduced_val,
            only_faiss=only_faiss,
            path_selection=path_selection,
            similarity_criterion=similarity_criterion,
            nearest_neighbors=nearest_neighbors,
            retrieval_config=retrieval_config,
            validation_batch_size=validation_batch_size,
        )

        if self.similarity_criterion == "multitask_retrieval":
            self.candidate_representation = str(
                self.retrieval_config.get("candidate_representation", "labels")
            )
            if self.candidate_representation not in {"labels", "predicted"}:
                raise ValueError(
                    "candidate_representation must be 'labels' or 'predicted'."
                )
            if self.candidate_representation == "predicted":
                logger.info(
                    "Predicting symmetric source and query retrieval representations"
                )
                fingerprints_train, fingerprints_test = (
                    self._refresh_multitask_predictions()
                )
            else:
                logger.info("Loading labeled source retrieval targets")
                (
                    self.candidate_fingerprints,
                    self.candidate_tasks,
                ) = self.get_candidate_retrieval_bank("train")
                self.train_tensor = torch.nn.functional.normalize(
                    self.candidate_fingerprints, p=2, dim=1
                )
                logger.info("Predicting multimodal retrieval signals for the test set")
                self.test_retrieval_predictions = self.get_multitask_predictions(
                    self._split_prediction_loader("test")
                )
                fingerprints_train = self.candidate_fingerprints
                fingerprints_test = self._retrieval_fingerprint_probabilities(
                    self.test_retrieval_predictions
                )
                self.test_tensor = torch.nn.functional.normalize(
                    torch.nan_to_num(fingerprints_test), p=2, dim=1
                )
            self.anchor_test_fingerprint_logits = (
                self.test_retrieval_predictions["fused_fingerprint_logits"]
                .detach()
                .cpu()
                .clone()
            )
            self.anchor_candidate_fingerprints = (
                self.candidate_fingerprints.detach().cpu().clone()
            )
            self.multitask_retriever = MultimodalCandidateRetriever(
                fingerprint_weight=float(
                    self.retrieval_config.get("fingerprint_weight", 1.0)
                ),
                task_weights=dict(
                    self.retrieval_config.get(
                        "task_weights", {"MSMS": 1.0, "NMR": 1.0, "IR": 1.0}
                    )
                ),
                anchor_weight=float(
                    self.retrieval_config.get("anchor_weight", 0.0)
                ),
                anchor_top_k_fraction=float(
                    self.retrieval_config.get("anchor_top_k_fraction", 0.0)
                ),
                formula_weight=float(
                    self.retrieval_config.get("formula_weight", 0.0)
                ),
                formula_similarity=str(
                    self.retrieval_config.get("formula_similarity", "cosine")
                ),
                reuse_penalty=float(
                    self.retrieval_config.get("reuse_penalty", 0.0)
                ),
            )
            if (
                self.multitask_retriever.anchor_weight > 0
                or self.multitask_retriever.anchor_top_k_fraction > 0
            ):
                logger.info(
                    "Enabled anchored retrieval current_weight={} "
                    "anchor_weight={} anchor_top_k_fraction={}",
                    1.0 - self.multitask_retriever.anchor_weight,
                    self.multitask_retriever.anchor_weight,
                    self.multitask_retriever.anchor_top_k_fraction,
                )
            self.candidate_formula_counts = None
            self.test_formula_counts = None
            if self.multitask_retriever.formula_weight != 0:
                formula_modality = str(
                    self.retrieval_config.get("formula_modality", "Formula")
                )
                for split in ("train", "test"):
                    if formula_modality not in self.dataset[split].column_names:
                        raise KeyError(
                            f"Formula modality {formula_modality!r} is absent from "
                            f"the {split!r} split."
                        )
                candidate_formulas = list(self.dataset["train"][formula_modality])
                test_formulas = list(self.dataset["test"][formula_modality])
                formula_counts, formula_elements = encode_formula_compositions(
                    candidate_formulas + test_formulas
                )
                n_candidates = len(candidate_formulas)
                self.candidate_formula_counts = formula_counts[:n_candidates]
                self.test_formula_counts = formula_counts[n_candidates:]
                logger.info(
                    "Enabled formula-aware retrieval weight={} similarity={} elements={}",
                    self.multitask_retriever.formula_weight,
                    self.multitask_retriever.formula_similarity,
                    formula_elements,
                )
            tensor = self.test_tensor
        else:
            # make embeddings (fps) train set
            logger.info('Making embeddings train set')
            fingerprints_train = self.get_fingerprints(self.datamodule.train_dataloader())
            tensor = torch.nn.functional.normalize(fingerprints_train, p=2, dim=1)
            tensor = tensor.detach().cpu()
            self.train_tensor = tensor

            # make embeddings (fps) test points
            logger.info('Making embeddings test set')
            fingerprints_test = self.get_fingerprints(self.datamodule.predict_dataloader())
            tensor = torch.nn.functional.normalize(fingerprints_test, p=2, dim=1)
            tensor = tensor.detach().cpu()
            self.test_tensor = tensor

        # save predicted fps
        if self.path_selection:
            df = self.dataset['train'].to_pandas()
            df['predicted_fingerprints'] = self.train_tensor.numpy().tolist()
            df['predicted_fingerprints_before_norm'] = fingerprints_train.detach().cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/train_tensor_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df

            df = self.dataset['test'].to_pandas()
            df['predicted_fingerprints'] = self.test_tensor.numpy().tolist()
            df['predicted_fingerprints_before_norm'] = fingerprints_test.detach().cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/test_tensor_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df
        
        del tensor, fingerprints_test, fingerprints_train
        
        if n_test_points is None or n_test_points <= 0:
            raise ValueError("n_test_points must be a positive integer.")
        self.n_test_points = int(n_test_points)
        self.n_train_points = (
            int(n_train_points)
            if n_train_points
            else max(1, int(self.batch_size / self.n_test_points))
        )

        # Perform k-means clustering with ncentroids and save the centroids
        logger.info(f'Clustering the test points in {n_clusters} clusters')
        kmeans = faiss.Kmeans(self.d_fp, n_clusters, niter=200, verbose=True, gpu=True, seed=seed)
        kmeans.train(self.test_tensor)

        # Calculate here the indices of the respective cluster for every test point
        D, I = kmeans.index.search(self.test_tensor, 1)
        self.I = I.reshape((len(I)))
        self.D = D

        self.clusters = self._build_clusters(
            np.asarray(kmeans.centroids), self.n_test_points
        )
        self.cluster_ids = sorted(self.clusters)
        self.num_ttt_clusters = len(self.cluster_ids)
        if not self.cluster_ids:
            raise ValueError(
                "No test cluster contains a sample with available retrieval modalities."
            )
        self.update_embeds = update_embeds
        # Track source-candidate exposure across TTT epochs.  The default
        # penalty is zero, preserving the original ranking; experiments can
        # enable a soft log-exposure penalty to reduce repeated candidates.
        self.selection_counts = torch.zeros(
            len(self.dataset["train"]), dtype=torch.float32
        )

        del kmeans

    def _build_clusters(
        self,
        centroids: np.ndarray,
        n_test_points: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Choose available representatives from within their assigned clusters."""
        clusters: Dict[int, Dict[str, Any]] = {}
        skipped: List[int] = []
        retrieval_availability = (
            self.test_retrieval_predictions.get("retrieval_availability")
            if self.similarity_criterion == "multitask_retrieval"
            and hasattr(self, "test_retrieval_predictions")
            else None
        )
        for cluster_value in sorted(set(int(value) for value in self.I.tolist())):
            member_indices = np.flatnonzero(self.I == cluster_value).astype(np.int64)
            representative_pool = member_indices
            if retrieval_availability is not None:
                representative_pool = np.asarray(
                    [
                        index
                        for index in member_indices
                        if bool(retrieval_availability[int(index)])
                    ],
                    dtype=np.int64,
                )
            if len(representative_pool) == 0:
                skipped.append(cluster_value)
                continue

            centroid = torch.as_tensor(
                centroids[cluster_value], dtype=self.test_tensor.dtype
            )
            candidate_vectors = self.test_tensor[representative_pool]
            distances = ((candidate_vectors - centroid) ** 2).sum(dim=1)
            closest_local = torch.argsort(distances)[: min(n_test_points, len(representative_pool))]
            representatives = representative_pool[closest_local.cpu().numpy()]
            clusters[cluster_value] = {
                "centroid": centroids[cluster_value],
                "indices": member_indices.tolist(),
                "ind_closest": representatives,
            }

        self.skipped_cluster_ids = skipped
        if skipped:
            logger.warning(
                "Skipping {} clusters without an available retrieval representative: {}",
                len(skipped),
                skipped,
            )
        return clusters
    
    def get_train_tensor(self):
        return self.train_tensor
    
    def get_test_tensor(self):
        return self.test_tensor

    def get_centroids(self):
        return {cluster_id: cluster["centroid"] for cluster_id, cluster in self.clusters.items()}
    
    def get_clusters(self):
        return self.clusters
    
    def get_clusters_info(self):
        clusters = self.get_clusters()

        for cluster_id in sorted(clusters.keys()):
            d_avg = np.mean(self.D[clusters[cluster_id]['indices']])
            d_max = np.max(self.D[clusters[cluster_id]['indices']])
            d_min = np.min(self.D[clusters[cluster_id]['indices']])
            logger.info(f"Cluster {cluster_id}: {len(clusters[cluster_id]['indices'])} points\nAvg distance = {d_avg}\n Max distance = {d_max}\n Min distance = {d_min}")

    def get_cluster_distances(self):
        return self.D

    def get_closest_points(self, centroid, n_points):
        index = faiss.IndexFlatIP(self.d_fp)
        index.add(self.test_tensor)
        _, indices = index.search(centroid, n_points)

        return self.test_tensor[indices[0]]

    # overwrite only the train dataloader
    def train_dataloader(self) -> DataLoader:

        # Update embeddings
        if self.update_embeds and self.update_embeds > 0:
            if self.epochs > 0 and self.epochs % self.update_embeds == 0:

                if self.similarity_criterion == "multitask_retrieval":
                    if self.candidate_representation == "predicted":
                        logger.info(
                            "Recomputing symmetric source and query retrieval predictions"
                        )
                        fingerprints_train, fingerprints_test = (
                            self._refresh_multitask_predictions()
                        )
                        self.anchor_candidate_fingerprints = (
                            self.candidate_fingerprints.detach().cpu().clone()
                        )
                    else:
                        logger.info(
                            "Recomputing multimodal predictions for the test set"
                        )
                        self.test_retrieval_predictions = self.get_multitask_predictions(
                            self._split_prediction_loader("test")
                        )
                        fingerprints_test = self._retrieval_fingerprint_probabilities(
                            self.test_retrieval_predictions
                        )
                        self.test_tensor = torch.nn.functional.normalize(
                            torch.nan_to_num(fingerprints_test), p=2, dim=1
                        )
                        fingerprints_train = self.candidate_fingerprints
                    self.anchor_test_fingerprint_logits = (
                        self.test_retrieval_predictions["fused_fingerprint_logits"]
                        .detach()
                        .cpu()
                        .clone()
                    )
                    tensor = self.test_tensor
                else:
                    logger.info("Recomputing embeddings/fingerprints for the training set")
                    fingerprints_train = self.get_fingerprints(self.datamodule.train_dataloader())
                    tensor = torch.nn.functional.normalize(fingerprints_train, p=2, dim=1)
                    tensor = tensor.detach().cpu()
                    self.train_tensor = tensor

                    logger.info("Recomputing embeddings/fingerprints for the test set")
                    fingerprints_test = self.get_fingerprints(self.datamodule.predict_dataloader())
                    tensor = torch.nn.functional.normalize(fingerprints_test, p=2, dim=1)
                    tensor = tensor.detach().cpu()
                    self.test_tensor = tensor

                # save predicted fps
                if self.path_selection:
                    df = self.dataset['train'].to_pandas()
                    df['predicted_fingerprints'] = self.train_tensor.numpy().tolist()
                    df['predicted_fingerprints_before_norm'] = fingerprints_train.detach().cpu().numpy().tolist()
                    with Path.open(Path(f'{self.path_selection}/train_tensor_iteration_{self.epochs}.json'), 'w') as f:
                        json.dump(df.to_json(), f)
                    f.close()
                    del df

                    df = self.dataset['test'].to_pandas()
                    df['predicted_fingerprints'] = self.test_tensor.numpy().tolist()
                    df['predicted_fingerprints_before_norm'] = fingerprints_test.detach().cpu().numpy().tolist()
                    with Path.open(Path(f'{self.path_selection}/test_tensor_iteration_{self.epochs}.json'), 'w') as f:
                        json.dump(df.to_json(), f)
                    f.close()
                    del df

        id_cluster = self.cluster_ids[self.epochs % len(self.cluster_ids)]
        logger.info(f"Training model for cluster id={id_cluster}")
        # Select the test point we want to use for selection in this cluster (we are already storing the index)
        ind_test_points = self.clusters[id_cluster]['ind_closest']
        test_points = self.test_tensor[ind_test_points] # in this way we update the representation consistently
                
        cleanup_faiss = False
        if self.similarity_criterion == 'fingerprints':
            indices = []
            cpu_index = faiss.IndexFlatIP(self.d_fp)  # inner product for cosine similarity
            cpu_index.add(self.train_tensor)

            # retrieval of datapoints using activeft code
            retriever = Retriever(cpu_index, fast=False, only_faiss=self.only_faiss, also_query_opposite=True) # we're using inner product index, so negative sim values should also be considered
            indices = []
            time_tot_faiss = 0
            time_tot_sift = 0
            for test_point in test_points:
                _, ind, _, time_retrieval = retriever.search(np.array([test_point]), N=self.n_train_points, K=self.nearest_neighbors, threads=self.num_workers)
                indices = np.concatenate((indices,ind), axis=None)
                time_tot_faiss += time_retrieval.faiss
                time_tot_sift += time_retrieval.sift
            indices_nopad = indices[indices >= 0] # type:ignore
            logger.info(f"Time taken for faiss selection = {time_tot_faiss}")
            logger.info(f"Additional time taken for sift selection = {time_tot_sift}")
            cleanup_faiss = True
        elif self.similarity_criterion == "multitask_retrieval":
            query_predictions = {
                "fused_fingerprint_logits": self.test_retrieval_predictions[
                    "fused_fingerprint_logits"
                ][ind_test_points],
                "task_logits": {
                    name: values[ind_test_points]
                    for name, values in self.test_retrieval_predictions[
                        "task_logits"
                    ].items()
                },
                "modality_weights": self.test_retrieval_predictions[
                    "modality_weights"
                ][ind_test_points],
                "modality_order": self.test_retrieval_predictions["modality_order"],
            }
            retrieval_result = self.multitask_retriever.rank(
                query_predictions,
                self.candidate_fingerprints,
                self.candidate_tasks,
                top_k=min(self.n_train_points, len(self.candidate_fingerprints)),
                anchor_fingerprint_logits=(
                    self.anchor_test_fingerprint_logits[ind_test_points]
                    if getattr(self, "anchor_test_fingerprint_logits", None)
                    is not None
                    else None
                ),
                anchor_candidate_fingerprints=getattr(
                    self, "anchor_candidate_fingerprints", None
                ),
                query_formula_counts=(
                    getattr(self, "test_formula_counts", None)[ind_test_points]
                    if getattr(self, "test_formula_counts", None) is not None
                    else None
                ),
                candidate_formula_counts=getattr(
                    self, "candidate_formula_counts", None
                ),
                candidate_selection_counts=getattr(self, "selection_counts", None),
            )
            valid_indices = retrieval_result.indices[retrieval_result.valid_mask].tolist()
            indices_nopad = np.asarray(
                list(dict.fromkeys(int(index) for index in valid_indices)), dtype=np.int64
            )
            indices = indices_nopad
            if len(indices_nopad) == 0:
                raise ValueError(
                    f"Multitask retrieval returned no source candidate for cluster {id_cluster}."
                )
        else:
            raise ValueError(f"Selection with similarity criterion {self.similarity_criterion} not implemented.")

        # memorize selected indices
        for ind in indices:
            self.model.set_sel_indices.add(ind)
        if len(indices_nopad) > 0 and hasattr(self, "selection_counts"):
            selected_counts = torch.bincount(
                torch.as_tensor(indices_nopad, dtype=torch.long),
                minlength=self.selection_counts.shape[0],
            ).to(dtype=self.selection_counts.dtype)
            self.selection_counts += selected_counts

        # Save selection
        if self.path_selection:
            df = self.dataset['train'].select(indices_nopad).to_pandas()
            df['predicted_fingerprints'] = self.train_tensor[indices_nopad].numpy().tolist()
            df['index'] = indices.astype(int)
            
            df_test = pd.DataFrame()
            df_test['index'] = ind_test_points.astype(int)
            df_test['predicted_fingerprints'] = test_points.numpy().tolist()
            for k in self.data_config.keys():
                df_test[k] = self.dataset['test'].select(ind_test_points)[k]
            
            with Path.open(Path(f'{self.path_selection}/sel_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df

            with Path.open(Path(f'{self.path_selection}/test_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df_test.to_json(), f)
            f.close()
            del df_test

        train_loader = DataLoader(
            self.dataset["train"].select(indices_nopad),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle = True if not isinstance(self.dataset["train"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.epochs += 1

        if cleanup_faiss:
            del cpu_index, retriever

        return train_loader
    
    def predict_dataloader(
        self,
    ) -> DataLoader:
        """It selects the points in the test set that are only part of the cluster in question."""

        logger.info(f'Predicting for {len(self.dataset["test"])} test points')

        test_loader = DataLoader(
            self.dataset["test"],
            collate_fn=self.collator,
            batch_size=self.batch_size if self.batch_size <= 64 else 64,
            shuffle=False if not isinstance(self.dataset["test"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader
