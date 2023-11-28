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
    ContentMatchTypeEnum,
    SimpleResponseLayoutCriteria,
    MatchContentCriteria,
    MatchLayoutCriteria,
    LayoutMatchTypeEnum
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
        Your task is to provide a clear and direct answer to the last question found in the text.        
        Maintain an objective tone by sticking to factual information and logical deductions without personal opinions or emotional language:
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
    # scope 1: bullet points, scope 2: numbered list, scope 3: simple layout
    scope = random.randint(1, 3)
    
    select_bullet_point_layout = scope == 1
    select_numbered_list_layout = scope == 2

    # scope 1 or 2: define criteria set for bullet points or numbered list
    if select_bullet_point_layout or select_numbered_list_layout:                
        if select_bullet_point_layout:
            layout_criteria = MatchLayoutCriteria(
                layout_type=LayoutMatchTypeEnum.UNORDERED_LIST,
                penalty=0.5,
                text="Your response should be ordered in format of bullet points.",
            )
        else:
            layout_criteria = MatchLayoutCriteria(
                layout_type=LayoutMatchTypeEnum.NUMBERED_LIST,
                penalty=0.5,                
            )

        possible_other_criterion = [
            MatchLengthCriteria(
                penalty=0.25,
                target_length=random.randint(100, 200),
                unit=TextLengthUnitEnum.WORDS,
            ),
            MatchLengthCriteria(
                penalty=0.25,
                target_length=random.randint(8, 12),
                unit=TextLengthUnitEnum.SENTENCES,
            ),
        ]    
    # scope 3: define criteria set for simple layout
    else:
        layout_criteria = SimpleResponseLayoutCriteria(penalty=0.5)

        possible_other_criterion = [
            MatchLengthCriteria(
                penalty=0.25,
                target_length=random.randint(50, 200),
                unit=TextLengthUnitEnum.WORDS,
            ),
            MatchLengthCriteria(
                penalty=0.25,
                target_length=random.randint(4, 8),
                unit=TextLengthUnitEnum.SENTENCES,
            ),
        ]

    random_sampled_criterion = random.sample(possible_other_criterion, 1)
    defined_criteria = [layout_criteria] + random_sampled_criterion

    return SummaryTask(
        base_text=base_text,
        criteria=defined_criteria,
        task_type="summarization",
        task_name="augment",
    )
  

def create_qg_task(base_text: str, index: int) -> QuestionGenerationTask:
    questions_prefixes = [
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "can",
        "do",
        "does",
        "did",
        "would",
        "could",
        "will",
        "shall",
        "may",
        "might",
        "am",
        "was",
        "were",
        "has",
        "have",
        "had",
        "been",
        "being",
    ]

    question_starts_with_prefix_criteria = MatchContentCriteria(
        contentMatchType=ContentMatchTypeEnum.STARTS_WITH,
        penalty=0.25,
        words_array=questions_prefixes,
        n_words=3,
    )

    question_ends_with_criteria = MatchContentCriteria(
        contentMatchType=ContentMatchTypeEnum.ENDS_WITH,
        penalty=0.25,
        words_array=["?"],
        n_words=1,
        text='Your response should end with a question mark, i.e. "?"',
    )

    other_random_criteria = [
        MatchLengthCriteria(
            penalty=0.25,
            target_length=random.randint(10, 40),
            unit=TextLengthUnitEnum.WORDS,
        ),
        MatchLengthCriteria(
            penalty=0.25,
            target_length=random.randint(40, 150),
            unit=TextLengthUnitEnum.CHARACTERS,
        ),
    ]

    random_sampled_criteria = random.sample(other_random_criteria, 1)
    criteria = [
        question_starts_with_prefix_criteria,
        question_ends_with_criteria,
    ] + random_sampled_criteria

    return QuestionGenerationTask(
        base_text=base_text,
        criteria=criteria,
        task_type="question-generation",
        task_name=f"followup{index}",
    )


def create_qa_task(base_text: str, index: int) -> QuestionAnswerTask:
    answer_should_not_include_criteria = MatchContentCriteria(
        words_array=["?"],
        n_words=1,
        penalty=0.2,
        contentMatchType=ContentMatchTypeEnum.INCLUDES,
        negate_match=True,
        text="Your response should not include any question marks",
    )

    simple_response_layout_criteria = SimpleResponseLayoutCriteria(penalty=0.2)

    words_criteria = MatchLengthCriteria(
        penalty=0.2,
        target_length=random.randint(50, 200),
        unit=TextLengthUnitEnum.WORDS,
    )

    criteria = [
        answer_should_not_include_criteria,
        simple_response_layout_criteria,
        words_criteria,
    ]

    return QuestionAnswerTask(
        base_text=base_text,
        criteria=criteria,
        task_type="question-answer",
        task_name=f"answer{index}",
    )
