from typing import Union, List, Optional, Dict, Mapping, Sized

import numpy as np
from torch import TensorType
from stripedhyena.tokenizer import CharLevelTokenizer
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput
from transformers.utils import PaddingStrategy, is_tf_tensor, is_torch_tensor, to_py_obj

from src.self_logger import logger


class EvoCharLevelTokenizer(CharLevelTokenizer, PreTrainedTokenizerBase):

    def __init__(self, vocab_size, name: str = "EvoCharLevelTokenizer", pad_token="N"):
        super(EvoCharLevelTokenizer, self).__init__(vocab_size=vocab_size)
        self.name = name
        self.pad_token = pad_token

    def __call__(
            self,
            batch_text_or_text_pairs: Union[
                TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
            **kwargs
    ):
        return self.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs, **kwargs)

    def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[List[str], str],
            padding: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[str] = None,
            **kwargs
    ):
        if isinstance(batch_text_or_text_pairs, list):
            max_seq_length = max([len(s) for s in batch_text_or_text_pairs])
            input_ids = []
            attention_mask = []
            token_type_ids = []
            for seq in batch_text_or_text_pairs:
                padding_list = [self.pad_id] * (max_seq_length - len(seq))
                mask = [1] * len(seq) + [0] * len(padding_list)
                attention_mask.append(mask)
                token_type_id = [0] * max_seq_length
                token_type_ids.append(token_type_id)
                input_id = self.tokenize(seq) + padding_list
                input_ids.append(input_id)
            encoded_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        else:
            encoded_inputs = {
                "input_ids": self.tokenize(text=batch_text_or_text_pairs),
                "attention_mask": [1] * len(batch_text_or_text_pairs),
                "token_type_ids": [0] * len(batch_text_or_text_pairs)
            }

        batch_outputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors
        )
        return batch_outputs

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if len(tokens) == 1:
            token_id = list(np.fromstring(tokens, dtype=np.uint8))[0]
        else:
            token_id = list(np.fromstring(tokens, dtype=np.uint8))
        return token_id

    def pad(
            self,
            encoded_inputs: Union[
                BatchEncoding,
                List[BatchEncoding],
                Dict[str, EncodedInput],
                Dict[str, List[EncodedInput]],
                List[Dict[str, EncodedInput]],
            ],
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = True,
    ) -> BatchEncoding:
        # Convert padding_strategy in PaddingStrategy
        if self.__class__.__name__.endswith("Fast"):
            if not self.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False):
                logger.warning_advice(
                    f"You're using a {self.__class__.__name__} tokenizer. Please note that with a fast tokenizer,"
                    " using the `__call__` method is faster than using a method to encode the text followed by a call"
                    " to the `pad` method to get a padded encoding."
                )
                self.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0):
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.uint8):
                return_tensors = "np" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.int64):
                return_tensors = "np" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.int32):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length
        )
        # import pdb
        # pdb.set_trace()
        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)
