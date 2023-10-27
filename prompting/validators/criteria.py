import re
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List

@dataclass
class TaskCriterion(ABC):
    """
    Abstract base class for defining task-specific evaluation criteria.

    Attributes:
        text (str): Text of the criterion to be added to the prompt.
        penalty (float): Penalty value associated with the criterion.
    Returns:
        torch.FloatTensor: Tensor containing the penalty values for each response.        
    """
    text: str
    penalty: float

    @abstractmethod
    def evaluate(self, completions: List[str]) -> torch.FloatTensor:
        pass

    @abstractmethod
    def compose_text(self) -> str:
        pass

@dataclass
class MaxOfNWordsCriteria(TaskCriterion):
    text: str = "Your response must contain at most {n_words} words."
    n_words : int = 50
    penalty: float = 0.1

    def evaluate(self, completions: List[str]) -> torch.FloatTensor:        
        penalties = torch.zeros(len(completions), dtype=torch.float32)
    
        for idx, response in enumerate(completions):
            surpass_n_words = len(re.split(r' +', response)) > self.n_words
            if surpass_n_words:
                penalties[idx] = self.penalty

        return penalties

    def compose_text(self) -> str:
        return self.text.format(n_words=self.n_words)
    
@dataclass
class MatchNSentencesCriteria(TaskCriterion):
    text: str = "Your response must have {n_sentences} sentences."
    n_sentences : int = 5
    penalty: float = 0.1

    def evaluate(self, completions: List[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(completions), dtype=torch.float32)

        for idx, response in enumerate(completions):
            response_n_sentences = len(re.split(r'\. +', response))
            if response_n_sentences != self.n_sentences:
                penalties[idx] = self.penalty

        return penalties

    def compose_text(self) -> str:            
        return self.text.format(n_sentences=self.n_sentences)
