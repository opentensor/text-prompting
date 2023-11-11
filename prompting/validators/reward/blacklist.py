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
import math
from fuzzywuzzy import fuzz
from typing import List, Union
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent
from transformers import BertTokenizer
from dataclasses import dataclass


# TODO: Use CLI arguments to set blacklist values: the most important being the boundary value and max_size


@dataclass
class BlacklistRewardEvent(BaseRewardEvent):
    matched_ngram: str = None
    significance_score: float = None


class Blacklist(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.blacklist.value

    def __init__(
        self,
        boundary: float = 6,
        n_min: int = 5,
        n_max: int = 14,
        word_limit: int = 2000,
        A: float = 1.3,
        preprocess: str = "[^(\\w|\\s)]",
        partial_ratio_boundary: float = 95,
        half_life: int = 20000,
        support: float = 0.01,
        error: float = 0.001,
        memory_lim: int = 1_000_000,
        frequency_multiplier: float = 100,
    ):
        """N-gram blacklist reward model which penalizes overused phrases in the network

        Args:
            boundary (float, optional): Cutoff for flagging completions and giving zero reward. Defaults to 6.
            max_size (int, optional): Maximum size of sliding window to use for aggregating ngrams. Defaults to 1_000_000.
            n_min (int, optional): Smallest ngram size. Defaults to 5.
            n_max (int, optional): Largest ngram size. Defaults to 14.
            word_limit (int, optional): Maximum word length, to prevent extremely long completions from overworking the queue. Defaults to 2000.
            A (float, optional): Exponent used in significance scoring, smaller A gives more weight to smaller ngrams. Values of 1.1-2 are recommended. Defaults to 1.3.
            preprocess (str, optional): Regex preprocessing string to make text more uniform. Defaults to '[^(\w|\s)]'.
            partial_ratio_boundry (int, optional): Boundry for fuzzy match. Default to 95.
            half_life (int, optional): Half life of the counter. ie. When the number of completions processed > half life, then put all the counters in half.
            support (float, optional): The percentage of times that a phrase need to appear to get the phrase kept in counter. (support should be >> counter)
            error (float, optional): Error parameter for lossy sampling, should be as small as possible, further decreasing it further will increase memory usage. (support should be >> error )
            memory_lim (int, optional): Max number of counter entry to save for memory protection.
            frequency_multiplier (float, optional): Multiplier for phrases frequency. Default to 100.
        """
        super().__init__()

        self.counter = {}

        self.n_min = n_min
        self.n_max = n_max
        self.word_limit = word_limit

        self.significance_scores = {}  # Store significance scores
        self.A = A
        self.boundary = boundary
        self.partial_ratio_boundary = partial_ratio_boundary

        self.preprocess = re.compile(preprocess) if preprocess else None
        self._last_update = 0

        # Lossy sampling parameters
        self.support = support
        self.error = error
        self.window = math.ceil(
            1 / self.error
        )  # Window size, counter would get pruned once for each window.
        self.w_current = 1  # window index.
        self.num_ngram = 0
        self.num_completion = 0

        self.half_life = half_life
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.memory_lim = memory_lim
        self.frequency_multiplier = frequency_multiplier

    def add(self, texts: List[str]):
        """Extract and add n-grams from a list of texts to counter

        Args:
            texts (list): batch of completion texts
        """

        for text in texts:
            # Extract n-grams from lowercased text
            ngrams = self.extract_ngrams(text.lower())

            if ngrams:
                self._add_ngrams(ngrams)

    def extract_ngrams(self, text: str) -> List[tuple]:
        """Extract n-grams from text string

        Args:
            text (str): completion text

        Returns:
            list: List of n-gram tuples

        """

        if self.preprocess:
            # remove all punctuation
            text = self.preprocess.sub("", text)

        words = self.tokenizer(text.lower())["input_ids"][1:-1]

        if self.word_limit is not None:
            words = words[: self.word_limit]

        ngrams = []
        for i in range(self.n_min, self.n_max + 1):
            ngrams.extend(zip(*[words[j:] for j in range(i)]))

        return ngrams

    def _add_ngrams(self, ngrams: List[tuple]):
        """Adds n-grams to counter, removing old n-grams periodically.
        Counting and pruning method based on Lossy counter.
        Reference: https://files.ifi.uzh.ch/dbtg/sdbs13/T01.3.pdf

        Args:
            ngrams (List[tuple]): List of n-gram tuples
        """

        for ngram in ngrams:
            if ngram in self.counter:
                self.counter[ngram][0] += 1
            else:
                # Store the tuple (frequency, max_error)
                self.counter[ngram] = [1, self.w_current - 1]

            self.num_ngram += 1

        self.num_completion += 1

        # Prune when move to next window.
        if self.num_completion % self.window == 0:
            self.w_current = math.ceil(self.num_completion / self.window)
            self.prune()

        # Safety feature: prune when reached max memory size.
        if len(self.counter) > self.memory_lim:
            self.w_current += 1
            self.prune()

        # Apply half life for the counter
        if self.num_completion > self.half_life:
            self.set_counter_to_half()

    def prune(self):
        """Prune the counter when the count is smaller then bucket index."""
        prune_ele = []
        for ele, (frequency, max_error) in self.counter.items():
            if frequency + max_error <= self.w_current:
                prune_ele.append(ele)

        for ele in prune_ele:
            del self.counter[ele]

    def reset(self):
        """Reset counters to initial values."""
        self.num_ngram = 0
        self.num_completion = 0
        self.w_current = 1
        self.counter = {}
        self.significance_scores = {}
        self._last_update = 0

    def calculate_significance(self) -> dict:
        """Calculate significance of all n-grams in counter. By construction, n-grams with count 1 will have significance 0.

        Returns:
            dict: Dictionary of n-gram tuples and their significance scores
        """

        significance_scores = {}
        for ngram, count in self.counter.items():
            if count[0] + count[1] > max(
                self.support * self.num_completion, self.w_current + 1
            ):
                decoded_ngram = self.tokenizer.decode(ngram)
                if len(decoded_ngram.split()) >= self.n_min:
                    # calculate significance score for ngram
                    significance_scores[decoded_ngram] = (
                        self.A ** (len(decoded_ngram) - 1)
                        * ((count[0] + count[1]) / self.num_completion)
                        * self.frequency_multiplier
                    )

        self._last_update = self.num_completion

        return dict(
            sorted(significance_scores.items(), key=lambda x: x[1], reverse=True)
        )

    def get_significance(self) -> dict:
        """Get significance scores, only recalculating if the counter has been updated.

        Returns:
            dict: Dictionary of n-gram tuples and their significance scores
        """

        if self.num_completion - self._last_update > self.window:
            self.significance_scores = self.calculate_significance()

        return self.significance_scores

    def most_common(self, n: int = 10) -> dict:
        """Get most common n-grams in queue

        Args:
            n (int): Number of most common n-grams to return. Defaults to 10.

        Returns:
            dict: Sorted dictionary of n-gram tuples and their counts
        """
        return sorted(
            self.counter.items(), key=lambda x: x[1][0] + x[1][1], reverse=True
        )[:n]

    def most_significant(self, n: int = 10, force_update: bool = True) -> dict:
        """Get most significant n-grams in queue based on significance scores

        Args:
            n (int, optional): Number of most significant n-grams to return. Defaults to 10.
            force_update (bool, optional): Force recalculate the significance scores. Defaults to True.

        Returns:
            dict: Sorted dictionary of n-gram tuples and their significance scores
        """

        scores = self.get_significance() if force_update else self.significance_scores
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def set_counter_to_half(self):
        """Set all the counters to half for a rolling window effect."""
        self.num_ngram = math.ceil(self.num_ngram / 2)
        self.num_completion = math.ceil(self.num_completion / 2)
        self.w_current = math.ceil(self.num_completion / self.window)
        self.counter = {
            tokens: [math.ceil(count[0] / 2), math.ceil(count[1] / 2)]
            for tokens, count in self.counter.items()
        }
        self._last_update = 0

    def reward(self, prompt: str, completion: str, name: str) -> BlacklistRewardEvent:
        """Reward function for blacklist reward model. Returns 1 if completion contains an n-gram with significance above the boundary, 0 otherwise.

        Args:
            prompt (str): Prompt text
            completion (str): Completion text
            name (str): Name of the validation step

        Returns:
            float: Reward value {0,1}
        """

        reward_event = BlacklistRewardEvent()

        if completion in prompt:
            reward_event.reward = 0.0
            return reward_event

        reward_event.reward = 1
        return reward_event

    def get_rewards(
        self, prompt: str, completions: List[str], name: str
    ) -> List[BlacklistRewardEvent]:
        # Get all the reward results.
        reward_events = [
            self.reward(prompt, completion, name) for completion in completions
        ]
        return reward_events

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards
