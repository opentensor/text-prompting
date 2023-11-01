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
import numpy as np
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

    def _count_sentences(self, text):
        # Regex explanation:
        # \b[A-Z]: Match an uppercase letter at the beginning of a word (assumed start of a sentence).
        # (?:[a-zA-Z]\. ){0,2}: Match 0-2 occurrences of an abbreviation-like pattern (single letter followed by a dot and space).
        # (?:[a-zA-Z]+\s+){1,}: Match at least one word followed by one or more whitespace characters.
        # [a-zA-Z]+: Match the last word before the end punctuation.
        # [.?!]: Match sentence-ending punctuation.
        # (?!\w): Negative lookahead to ensure the punctuation is not followed by an alphanumeric character (part of an abbreviation).
        # The pattern ignores common abbreviations by not counting them as sentence terminators.
        pattern = r"\b[A-Z](?:[a-zA-Z]\. ){0,2}(?:[a-zA-Z]+\s+){1,}[a-zA-Z]+[.?!](?!\w)"

        # Find all matches
        sentences = re.findall(pattern, text)

        # Return the count
        return len(sentences)

    def _get_completion_length(self, response: str) -> int:
        unit_to_split_pattern = {
            TextLengthUnitEnum.CHARACTERS: None,
            TextLengthUnitEnum.SENTENCES: None,
            TextLengthUnitEnum.WORDS: r"\s+",
            TextLengthUnitEnum.PARAGRAPHS: r"\n\n+",
        }

        if self.unit == TextLengthUnitEnum.CHARACTERS:
            return len(response)
        elif self.unit == TextLengthUnitEnum.SENTENCES:
            return self._count_sentences(response)
        else:
            split_pattern = unit_to_split_pattern[self.unit]
            return len(re.split(split_pattern, response.strip()))

    def evaluate(self, completions: List[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(completions), dtype=torch.float32)
        for idx, completion in enumerate(completions):
            completion_length = self._get_completion_length(completion)
            if completion_length != self.target_length:
                # Scales the penalty using a quadratic function based on the deviation of the response length from the target length.
                # The penalty is designed to be gentler on smaller deviations and steeper on larger deviations from the target length.
                # The computed penalty is capped between 0 and 1.
                penalty_scale_factor = np.abs(
                    1 - (completion_length / self.target_length) ** 2
                ).clip(0, 1)

                scaled_penalty = self.penalty * penalty_scale_factor
                penalties[idx] = scaled_penalty

        return penalties

    def compose_text(self) -> str:
        return self.text.format(target_length=self.target_length, unit=self.unit.value)
