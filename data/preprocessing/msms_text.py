import logging
from dataclasses import dataclass, field
from typing import List, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from ..tokenizer import build_regex_tokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MSMSTextPreprocessor:
    tokenizer: AutoTokenizer = field(init=False)
    max_sequence_length: int = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        msms_spectra = sampled_dataset[modality]
        processed_msms = self.process_msms(msms_spectra)
        
        longest_sequence = max(processed_msms, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 15

        self.tokenizer = build_regex_tokenizer(
            processed_msms,
            regex_string="(\s)",
            tokenizer_behaviour="removed",
            max_length=self.max_sequence_length,
        )

        logging.info(f'Set max_sequence_length to {self.max_sequence_length}')

    def __call__(
        self, msms_spectra: List[Union[str, List[List[float]]]]
    ) -> torch.Tensor:
        processed_msms = self.process_msms(msms_spectra)

        tokenized_input = self.tokenizer(
            processed_msms,
            padding="longest",
            # SDBS real spectra can be longer than the source-domain length.
            # Padding is batch-local; the model's positional limit is checked
            # at the multimodal sequence level by the training configuration.
            truncation=False,
            return_tensors="pt",
        )
        no_data_mask = [not self._has_numeric_peak(msms) for msms in msms_spectra]
        tokenized_input["attention_mask"][no_data_mask] = 0

        return tokenized_input

    @staticmethod
    def _has_numeric_peak(msms: Union[str, List[List[float]], None]) -> bool:
        if msms is None:
            return False
        if not isinstance(msms, str):
            return bool(msms)
        for token in msms.split():
            try:
                float(token)
                return True
            except ValueError:
                continue
        return False

    def process_msms(
        self, msms_spectra: List[Union[str, List[List[float]], None]]
    ) -> List[str]:
        processed_msms = list()

        for msms in msms_spectra:
            if msms is None:
                processed_msms.append("")
                continue
            if isinstance(msms, str):
                processed_msms.append(msms.strip())
                continue
            msms_string = ""
            for peak in msms:
                if peak[1] < 1:
                    continue
                msms_string = msms_string + f"{round(peak[0], 1):.1f} {round(peak[1], 1):.1f} "

            msms_string = msms_string.strip()
            processed_msms.append(msms_string)

        return processed_msms
