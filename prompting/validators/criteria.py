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
from dataclasses import dataclass, field
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
        # Define str pattern to match
        pattern = r"(?<![A-Z])[\.\?!](?:\s|$)"

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
                # Computes the relative error as the deviation of the response length from the target length, normalized by the target length.
                # Scales the penalty using an exponential function based on this relative error.
                # The penalty starts off small for minor deviations but increases rapidly for larger deviations.
                # The formula ensures that the penalty lies between 0 and 1.
                relative_error = (
                    self.target_length - completion_length
                ) / self.target_length
                penalty_scale_factor = 1 - np.exp(-10 * relative_error**2)

                scaled_penalty = self.penalty * penalty_scale_factor
                penalties[idx] = scaled_penalty

        return penalties

    def compose_text(self) -> str:
        return self.text.format(target_length=self.target_length, unit=self.unit.value)


# Enum for Content Match Type
class ContentMatchTypeEnum(Enum):
    STARTS_WITH = "starts with"
    ENDS_WITH = "ends with"
    INCLUDES = "includes"


# The MatchContentCriteria class
@dataclass
class MatchContentCriteria(TaskCriterion):
    default_text: str = "Your response should {match_type} the following words: {words}."
    text: str = default_text
    penalty: float = 0.1
    n_words: int = 3
    words_array: List[str] = field(default_factory=list)
    contentMatchType: ContentMatchTypeEnum = ContentMatchTypeEnum.STARTS_WITH
    sampled_words: List[str] = field(init=False)
    negate_match: bool = False

    def __post_init__(self):
        # Randomly sample words from the array based on n_words
        self.sampled_words = np.random.choice(self.words_array, self.n_words, replace=False)

    def _get_regex_pattern(self):
        # Escape all special characters in the sampled words
        escaped_words = map(re.escape, self.sampled_words)

        if self.contentMatchType == ContentMatchTypeEnum.STARTS_WITH:
            return rf"^\s*({'|'.join(escaped_words)})\b"
        elif self.contentMatchType == ContentMatchTypeEnum.ENDS_WITH:            
            return rf"({'|'.join(escaped_words)})\s*$"
        else:  # ContentMatchTypeEnum.INCLUDES
            return rf"({'|'.join(escaped_words)})"

    def evaluate(self, completions: List[str]) -> torch.FloatTensor:
        penalties = torch.zeros(len(completions), dtype=torch.float32)
        # Define regex pattern based on contentMatchType
        pattern = self._get_regex_pattern()

        for idx, completion in enumerate(completions):
            # Check if the completion matches the pattern
            match = re.search(pattern, completion, re.IGNORECASE)
                        
            completion_with_undesired_match = self.negate_match and match
            completion_without_desired_match = not self.negate_match and not match

            if completion_with_undesired_match or completion_without_desired_match:
                penalties[idx] = self.penalty            

        return penalties

    def compose_text(self) -> str:
        # Check if the text property is different than the default. If so, use that text.
        if self.text != MatchContentCriteria.default_text:
            return self.text

        # Adds "should" or "should not" instruction based on the negate_match property
        should_match_text = "should" if not self.negate_match else "should not"

        # Get the list of selected sampled words
        words_list = ', '.join(self.sampled_words)

         # Get the descriptive text of the match type
        match_type_text = self.contentMatchType.value 

        # Adjust the text based on the number of words
        if self.n_words > 1:
            text = f"Your response {should_match_text} {match_type_text} one of the following words: {words_list}."
        else:
            text = f"Your response {should_match_text} {match_type_text} the following word: {words_list}."
        return text





