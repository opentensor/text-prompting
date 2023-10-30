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
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType


class SentenceMatchPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.sentence_length_penalty.value

    def calculate_penalties(
        self, task: Task, completions: List[str]
    ) -> torch.FloatTensor:
        # NOTE: This is an example placeholder, the data source can be easily expanded to include more sentences
        # or be externalized in a public hugging face dataset.
        penalizing_sentences = [
            "what have we learned from this task?",
            "here is a task",
            "here is the solution",
            "use complete sentences",
            # ...
        ]

        penalties = []
        for completion in completions:
            accumulated_penalty = 0.0
            for sentence in penalizing_sentences:
                if sentence in completion.lower():
                    accumulated_penalty += 0.1

            penalties.append(accumulated_penalty)

        return torch.tensor(penalties, dtype=torch.float32)
