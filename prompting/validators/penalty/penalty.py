import torch
import bittensor as bt
from enum import Enum
from typing import List
from abc import ABC, abstractmethod
from prompting.validators.tasks import Task

class BasePenaltyModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def max_penalty(self) -> float:
        ...

    @abstractmethod
    def calculate_penalties(task: Task, completions: List[str]) -> torch.FloatTensor:
        ...

    def apply_penalties(self, responses: List[bt.Synapse], task: Task) -> torch.FloatTensor:
        completions = [response.completion for response in responses]
        penalties = self.calculate_penalties(task, completions)

        # Cap values greater than max_penalty
        mask_greater_than_max_penalty = penalties > self.max_penalty
        penalties[mask_greater_than_max_penalty] = self.max_penalty

        return penalties    

class PenaltyModelType(Enum):
    task_validation_penalty = "task_validation_penalty"
    keyword_match_penalty = "keyword_match_penalty"
    sentence_length_penalty = "sentence_length_penalty"