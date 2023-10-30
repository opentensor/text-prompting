import torch
import bittensor as bt
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from prompting.validators.tasks import Task


class BasePenaltyModel(ABC):
    def __init__(self, max_penalty: float):
        self.max_penalty = max_penalty

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return str(self.name)

    @abstractmethod
    def calculate_penalties(task: Task, completions: List[str]) -> torch.FloatTensor:
        ...

    def apply_penalties(
        self, responses: List[bt.Synapse], task: Task
    ) -> torch.FloatTensor:
        completions = [response.completion for response in responses]
        raw_penalties = self.calculate_penalties(task, completions)

        # Copy raw penalties to a new tensor for adjustments
        adjusted_penalties = raw_penalties.clone()

        # If accumulated penalties are bigger than 1, set them to 1
        mask_greater_or_equal_1 = adjusted_penalties >= 1
        adjusted_penalties[mask_greater_or_equal_1] = 1

        # Adjust values greater than max_penalty
        mask_greater_than_max_penalty = adjusted_penalties > self.max_penalty
        adjusted_penalties[mask_greater_than_max_penalty] = self.max_penalty

        # Invert penalties to scale rewards accordingly
        applied_penalties = 1 - adjusted_penalties

        return raw_penalties, adjusted_penalties, applied_penalties


class PenaltyModelType(Enum):
    task_validation_penalty = "task_validation_penalty"
    keyword_match_penalty = "keyword_match_penalty"
    sentence_length_penalty = "sentence_length_penalty"
