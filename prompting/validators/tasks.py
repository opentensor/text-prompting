# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
import textwrap
import random
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List
from prompting.validators.criteria import (
    TaskCriterion,
    MatchLengthCriteria,
    TextLengthUnitEnum,
)


@dataclass
class Task(ABC):
    base_text: str
    task_name: str
    task_type: str
    criteria: List[TaskCriterion] = field(default_factory=list)

    @abstractmethod
    def compose_prompt(self) -> str:
        ...


class SummaryTask(Task):
    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [
            f"- {criterion.compose_text()}" for criterion in self.criteria
        ]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent(
            """\
        Your task is to summarize the text delimited with triple backticks:
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}
        - Do not try to create questions or answers for your summarization. 
        """
        )

        prompt = prompt_template.format(
            base_text=self.base_text, criteria=criteria_bullet_points_str
        )
        return prompt


class QuestionGenerationTask(Task):
    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [
            f"- {criterion.compose_text()}" for criterion in self.criteria
        ]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent(
            """\
        Your task is to ask a single relevant and insightful question about the preceding context delimited with triple backticks:
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}                                      
        - Do not answer the question you generate. 
        - Do not try to summarize the text
        """
        )

        prompt = prompt_template.format(
            base_text=self.base_text, criteria=criteria_bullet_points_str
        )
        return prompt


class QuestionAnswerTask(Task):
    def compose_prompt(self) -> str:
        # Aggregates criteria in bullet points
        criteria_bullet_points = [
            f"- {criterion.compose_text()}" for criterion in self.criteria
        ]
        criteria_bullet_points_str = "\n".join(criteria_bullet_points)

        prompt_template = textwrap.dedent(
            """\
        Read the preceding context delimited with triple backticks carefully.
        Your task is to provide a step-by-step answer to the last question found in the text, elaborating on your thought process:
        '''{base_text}'''
        
        The following criteria must be respected:
        {criteria}                                      
        - Do not include questions or summaries in your answer.
        """
        )

        prompt = prompt_template.format(
            base_text=self.base_text, criteria=criteria_bullet_points_str
        )
        return prompt


def create_summarization_task(base_text: str) -> SummaryTask:
    possible_criterias = [
        MatchLengthCriteria(penalty=0.1, target_length=random.randint(50, 200), unit=TextLengthUnitEnum.WORDS),
        MatchLengthCriteria(penalty=0.1, target_length=random.randint(4, 8),unit=TextLengthUnitEnum.SENTENCES)
    ]

    sampled_criterias = random.sample(possible_criterias, 1)

    return SummaryTask(
        base_text=base_text,
        criteria=sampled_criterias,
        task_type="summarization",
        task_name="augment",
    )


def create_qg_task(base_text: str, index: int) -> QuestionGenerationTask:
    possible_criterias = [
        MatchLengthCriteria(penalty=0.1, target_length=random.randint(25, 50), unit=TextLengthUnitEnum.WORDS),
        MatchLengthCriteria(penalty=0.1, target_length=random.randint(125, 250), unit=TextLengthUnitEnum.CHARACTERS)
    ]

    sampled_criterias = random.sample(possible_criterias, 1)

    return QuestionGenerationTask(
        base_text=base_text,
        criteria=sampled_criterias,
        task_type="question-generation",
        task_name=f"followup{index}",
    )


def create_qa_task(base_text: str, index: int) -> QuestionAnswerTask:
    possible_criterias = [
        MatchLengthCriteria(penalty=0.1,target_length=random.randint(50, 200),unit=TextLengthUnitEnum.WORDS),
        MatchLengthCriteria(penalty=0.1,target_length=random.randint(4, 8),unit=TextLengthUnitEnum.SENTENCES)
    ]

    sampled_criterias = random.sample(possible_criterias, 1)

    return QuestionAnswerTask(
        base_text=base_text,
        criteria=sampled_criterias,
        task_type="question-answer",
        task_name=f"answer{index}",
    )
