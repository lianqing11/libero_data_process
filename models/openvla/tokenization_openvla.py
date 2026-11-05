from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class OpenVLAActionTokenizer:
    """
    Implements the discretisation strategy used by OpenVLA.  Continuous actions in ``[-1, 1]``
    (or a custom interval) are uniformly quantised and mapped to *unused* linguistic tokens
    near the end of the base tokenizer's vocabulary.  The resulting discrete ids are later
    decoded back into bin centres to obtain continuous actions again.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ) -> None:
        if bins < 2:
            raise ValueError("`bins` must be >= 2.")
        if max_action <= min_action:
            raise ValueError("`max_action` must be greater than `min_action`.")

        self.tokenizer = tokenizer
        self.n_bins = int(bins)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        # Construct uniform bins and helper centres
        self.bins = np.linspace(self.min_action, self.max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _clip_and_digitize(self, action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        arr = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
        arr = np.clip(arr, a_min=self.min_action, a_max=self.max_action)
        return np.digitize(arr, self.bins)

    def to_token_ids(self, action: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert continuous actions into token ids (without decoding back to text)."""
        discretized = self._clip_and_digitize(action)
        return (self.tokenizer.vocab_size - discretized).astype(np.int64)

    def __call__(self, action: Union[np.ndarray, torch.Tensor]) -> Union[str, List[str]]:
        """
        Returns the literal decoded string corresponding to the action tokens so that the
        sequences can be concatenated directly inside textual prompts (mirrors the reference
        OpenVLA implementation).
        """
        token_ids = self.to_token_ids(action)
        if token_ids.ndim == 1:
            return self.tokenizer.decode(token_ids.tolist())
        flat = token_ids.reshape(token_ids.shape[0], -1).tolist()
        return self.tokenizer.batch_decode(flat)

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------
    def decode_token_ids_to_actions(self, action_token_ids: Union[np.ndarray, Iterable[int]]) -> np.ndarray:
        """
        Reconstruct continuous actions from discrete ids by mapping into the bin centres.
        """
        discretized_actions = self.tokenizer.vocab_size - np.asarray(action_token_ids, dtype=np.int64)
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
