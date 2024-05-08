"""
Implements a bridge between poli black boxes and the way in which PR expects objectives.

(Pattern-matching off of pest.py)
"""

import json
from typing import List, Optional, Dict, OrderedDict

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from torch import Tensor
from botorch.utils.torch import BufferDict

from discrete_mixed_bo.problems.base import (DiscreteTestProblem, MultiObjectiveTestProblem,
                                             DiscretizedBotorchTestProblem, BaseTestProblem)


class PoliObjective(DiscreteTestProblem):
    """
    A bridge between poli black boxes and PR.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        sequence_length: int,
        alphabet: List[str] = None,
        noise_std: float = None,
        negate: bool = False,
    ) -> None:
        self._bounds = [(0, len(alphabet) - 1) for _ in range(sequence_length)]
        self.dim = sequence_length
        self.black_box = black_box
        alphabet = alphabet or self.black_box.info.alphabet
        if alphabet is None:
            raise ValueError("Alphabet must be provided.")

        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        super().__init__(noise_std, negate, categorical_indices=list(range(self.dim)))

    def evaluate_true(self, X: torch.Tensor):
        # Evaluate true seems to be expecting
        # a tensor of integers.
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # 1. transform to a list of strings
        x_str = [[self.alphabet_i_to_s[int(i)] for i in x_i] for x_i in X.numpy(force=True)]

        # 2. evaluate the black box
        return torch.from_numpy(self.black_box(np.array(x_str)))


class PoliMultiObjective(DiscreteTestProblem, MultiObjectiveTestProblem):
    """
    A bridge between poli black boxes and PR.
    """
    num_objectives: int
    _ref_point: List[float]
    _discrete_values: dict
    _bounds: list

    def __init__(
        self,
        black_box: AbstractBlackBox,
        sequence_length: int,
        alphabet: List[str] = None,
        noise_std: float = None,
        negate: bool = False,
        integer_indices = None,
        integer_bounds = None,
        ref_point: List[float] = None,
    ) -> None:
        self._bounds = [(0, len(alphabet) - 1) for _ in range(sequence_length)]
        self.dim = sequence_length
        self.black_box = black_box
        alphabet = alphabet or self.black_box.info.alphabet
        self._ref_point = ref_point  #  NOTE: this assumes maximization of all objectives.
        self.num_objectives = ref_point.shape[-1]
        if alphabet is None:
            raise ValueError("Alphabet must be provided.")

        self.alphabet_s_to_i = {s: i for i, s in enumerate(alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(alphabet)}
        MultiObjectiveTestProblem.__init__(
            self,
            noise_std=noise_std,
            negate=negate,
        )
        self._setup(integer_indices=integer_indices)
        self.discrete_values = BufferDict()
        self._discrete_values = {f"pos_{i}": list(self.alphabet_s_to_i.values()) for i in range(sequence_length)}
        # for k, v in self._discrete_values.items():
        #     self.discrete_values[k] = torch.tensor(v, dtype=torch.float)
        #     self.discrete_values[k] /= self.discrete_values[k].max()
        # super().__init__(noise_std, negate, categorical_indices=list(range(self.dim)))
        for v in self._discrete_values.values():
            self._bounds.append((0, len(alphabet)))

    def evaluate_true(self, X: torch.Tensor):
        # Evaluate true seems to be expecting
        # a tensor of integers.
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # 1. transform to a list of strings
        x_str = [[self.alphabet_i_to_s[int(i)] for i in x_i] for x_i in X.numpy(force=True)]

        # 2. evaluate the black box
        return torch.from_numpy(self.black_box(np.array(x_str)))





if __name__ == "__main__":
    from poli.repository import AlbuterolSimilarityBlackBox

    f = AlbuterolSimilarityBlackBox(string_representation="SELFIES")
    with open(
        "/Users/sjt972/Projects/high_dimensional_bo_benchmark/data/small_molecule_datasets/processed/zinc250k_alphabet_stoi.json"
    ) as fp:
        alphabet = json.load(fp)

    alphabet = list(alphabet.keys())

    objective = PoliObjective(f, sequence_length=64, alphabet=alphabet)
