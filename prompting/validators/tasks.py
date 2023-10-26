import re
import torch
import textwrap
from dataclasses import dataclass, field
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
    def evaluate(self, responses: List[str]) -> torch.FloatTensor:
        pass

    @abstractmethod
    def compose_text(self) -> str:
        pass

@dataclass
class MaxOfNWordsCriteria(TaskCriterion):
    text: str = "Your response must contain at most {n_words} words."
    n_words : int = 50
    penalty: float = 0.1

    def evaluate(self, responses: List[str]) -> torch.FloatTensor:        
        penalties = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
    
        for idx, response in enumerate(responses):
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

    def evaluate(self, responses: List[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(responses), dtype=torch.float32).to(self.device)

        for idx, response in enumerate(responses):
            response_n_sentences = len(re.split(r'\. +', response))
            if response_n_sentences != self.n_sentences:
                penalties[idx] = self.penalty

        return penalties

    def compose_text(self) -> str:            
        return self.text.format(n_sentences=self.n_sentences)




@dataclass
class Task(ABC):
    """
    Abstract base class for defining a task with multiple evaluation criteria.

    Attributes:
        prompt (str): Question or instruction associated with the task.
        criteria (List[TaskCriterion]): List of evaluation criteria associated with the task.
        device (torch.device): Torch device where calculations will be performed. Default is CPU.    
    """
    base_prompt: str    
    criteria: List[TaskCriterion] = field(default_factory=list)   
    
    @abstractmethod
    def validate(self, responses: List[str]) -> torch.FloatTensor:
        """
        Abstract method to validate responses based on the task's criteria.

        Args:
            responses (List[str]): List of response strings to validate.

        Returns:
            torch.FloatTensor: Tensor containing the accumulated penalties for each response.

        Workflow:
            For each criterion in the task's criteria, it evaluates the responses and accumulates the penalties.
        """
        accumulated_penalties: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32)

        # Accumulate penalties for each criterion        
        for criterion in self.criteria:
            accumulated_penalties.add_(criterion.evaluate(responses))
            
        return accumulated_penalties

    @abstractmethod
    def compose_prompt(self) -> str:
        pass

        

class SummaryTask(Task):    
    def validate(self, responses: List[str]) -> torch.FloatTensor:        
        # The logic for validation is the same as the base class, so we can call the base class's method.
        return super().validate(responses)

    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [f"- {criterion.compose_text()}" for criterion in self.criteria]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent("""\
        Your task is to summarize the text delimited with triple backticks:
        '''{base_prompt}'''
        
        The following criteria should be respected:
        {criteria}
        - Do not try to create questions or answers for your summarization. 
        """)

        prompt = prompt_template.format(base_prompt=self.base_prompt, criteria=criteria_bullet_points_str)

        
        return prompt
