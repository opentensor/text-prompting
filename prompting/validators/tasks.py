import torch
import textwrap
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List
from prompting.validators.criteria import TaskCriterion, MaxOfNWordsCriteria, MatchNSentencesCriteria

@dataclass
class Task(ABC):
    base_text: str    
    task_name: str
    task_type: str
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
            Then, it updates the accumulated penalties based on conditions.
        """
        accumulated_penalties: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32)
        completions = [response.completion for response in responses]
        # Accumulate penalties for each criterion        
        for criterion in self.criteria:            
            accumulated_penalties.add_(criterion.evaluate(completions))

        # Update accumulated_penalties based on conditions
        mask_less_than_1 = accumulated_penalties < 1
        mask_greater_or_equal_1 = accumulated_penalties >= 1
        accumulated_penalties[mask_less_than_1] = 1 - accumulated_penalties[mask_less_than_1]
        accumulated_penalties[mask_greater_or_equal_1] = 0
            
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
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}
        - Do not try to create questions or answers for your summarization. 
        """)

        prompt = prompt_template.format(base_text=self.base_text, criteria=criteria_bullet_points_str)        
        return prompt

class QuestionGenerationTask(Task):
    def validate(self, responses: List[str]) -> torch.FloatTensor:
        # The logic for validation is the same as the base class, so we can call the base class's method.
        return super().validate(responses)

    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [f"- {criterion.compose_text()}" for criterion in self.criteria]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent("""\
        Your task is to ask a single relevant and insightful question about the preceding context delimited with triple backticks:
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}                                      
        - Do not answer the question you generate. 
        - Do not try to summarize the text
        """)

        prompt = prompt_template.format(base_text=self.base_text, criteria=criteria_bullet_points_str)        
        return prompt

class QuestionAnswerTask(Task):
    def validate(self, responses: List[str]) -> torch.FloatTensor:
        # The logic for validation is the same as the base class, so we can call the base class's method.
        return super().validate(responses)
        
    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [f"- {criterion.compose_text()}" for criterion in self.criteria]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent("""\
        Read the preceding context delimited with triple backticks carefully. Your task is to answer the question step by step and explain your thoughts:
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}                                      
        - Do not include questions or summaries in your answer.
        """)

        prompt = prompt_template.format(base_text=self.base_text, criteria=criteria_bullet_points_str)        
        return prompt


def create_summarization_task(base_text: str) -> SummaryTask:
    criteria = [
        MaxOfNWordsCriteria(n_words=random.randint(50, 200), penalty=0.1),
        MatchNSentencesCriteria(n_sentences=random.randint(4, 8), penalty=0.1)
    ]

    return SummaryTask(base_text=base_text, criteria=criteria, task_type='summarization', task_name='augment')

def create_qg_task(base_text: str, index: int) -> QuestionGenerationTask:
    criteria = [
        MaxOfNWordsCriteria(n_words=random.randint(25, 40), penalty=0.1),
    ]

    return QuestionGenerationTask(base_text=base_text, criteria=criteria, task_type='question-generation', task_name=f'followup{index}')

def create_qa_task(base_text: str, index:int) -> QuestionAnswerTask:
    criteria = [
        MaxOfNWordsCriteria(n_words=random.randint(50, 200), penalty=0.1),
        MatchNSentencesCriteria(n_sentences=random.randint(4, 8), penalty=0.1)
    ]

    return QuestionAnswerTask(base_text=base_text, criteria=criteria, task_type='question-answer', task_name=f'answer{index}')