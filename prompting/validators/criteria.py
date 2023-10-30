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
import re
import torch
import json
import yaml
import ast
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
from enum import Enum


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


class TextLengthUnitEnum(Enum):
    CHARACTERS = "characters"
    WORDS = "words"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"


@dataclass
class MatchLengthCriteria(TaskCriterion):
    text: str = "Your response must have {target_length} {unit}."
    penalty: float = 0.1
    target_length: int = 100
    unit: TextLengthUnitEnum = TextLengthUnitEnum.WORDS

    def _get_completion_length(self, response: str) -> int:
        unit_to_split_pattern = {
            TextLengthUnitEnum.CHARACTERS: None,
            TextLengthUnitEnum.WORDS: r" +",
            TextLengthUnitEnum.SENTENCES: r"\. +",
            TextLengthUnitEnum.PARAGRAPHS: r"\n\n+",
        }

        if self.unit == TextLengthUnitEnum.CHARACTERS:
            return len(response)
        else:
            split_pattern = unit_to_split_pattern[self.unit]
            return len(re.split(split_pattern, response.strip()))

    def evaluate(self, completions: List[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(completions), dtype=torch.float32)
        for idx, completion in enumerate(completions):
            completion_length = self._get_completion_length(completion)
            if completion_length != self.target_length:
                # Scales the penalty based on how close the response length is to the expected target length.
                # If the response length is closer to the target, the penalty is reduced, ensuring that
                # small deviations from the target length are penalized less than larger deviations.
                penalty_scale_factor = min(
                    abs(1 - (completion_length / self.target_length)), 1
                )

                scaled_penalty = self.penalty * penalty_scale_factor
                penalties[idx] = scaled_penalty

        return penalties

    def compose_text(self) -> str:
        return self.text.format(target_length=self.target_length, unit=self.unit.value)
    
class LayoutTypeEnum(Enum):
    JSON = "json"
    YAML = "yaml"
    DICTIONARY = "python dictionary"
    NUMBEREDLIST = "numbered list"
    BULLETPOINTLIST = "bullet point list"

@dataclass
class MatchLayoutCriteria(TaskCriterion):
    text: str = "Your response must be in the form of a {format_type}{w}{fields}"
    penalty: float = 0.1
    format_type: LayoutTypeEnum = LayoutTypeEnum.JSON
    num_fields: int = 0
    fields : str = " with {num_fields} fields"

    def is_json(text):
        try:
            json.loads(text)
            return True
        except ValueError:
            return False
    
    def is_yaml(text):
        try:
            yaml.safe_load(text)
            return True
        except yaml.YAMLError:
            return False
        
    def is_dictionary(text):
        try:
            if type(ast.literal_eval(text)) == dict:
                return True
            else:
                return False
        except (ValueError, SyntaxError):
            return False

    def is_numbered_list(input_string):
        pattern = r'^\d.*\n?$'
        lines = input_string.split('\n')
        return all(re.match(pattern, line) for line in lines)
    
    def is_bullet_point_list(input_string):
        pattern = r'^\s*[-*+]\s.*\n?(\s*[-*+]\s.*\n?)*$'
        return bool(re.match(pattern, input_string))

    def _get_format_match(self, response : str) -> bool:
        if self.format_type == LayoutTypeEnum.JSON:
            return self.is_json(response)
        elif self.format_type == LayoutTypeEnum.YAML:
            return self.is_yaml(response)
        elif self.format_type == LayoutTypeEnum.DICTIONARY:
            return self.is_dictionary(response)
        elif self.format_type == LayoutTypeEnum.NUMBEREDLIST:
            return self.is_numbered_list(response)
        elif self.format_type == LayoutTypeEnum.BULLETPOINTLIST:
            return self.is_bullet_point_list(response)
        else:
            return False
        
    def evaluate(self, completions: list[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(completions), dtype = torch.float32)
        for idx, completion in enumerate(completions):
            if not self._get_format_match(completion):
                penalties[idx] = self.penalty
        return penalties

    def compose_text(self) -> str:
        if self.num_fields == 0:
            return self.text.format(format_type = self.format_type.value, fields = "")
        return self.text.format(format_type = self.format_type.value, fields = self.fields)
