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
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType


class ContentMatchPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.sentence_match_penalty.value

    def calculate_penalties(
        self, task: Task, completions: List[str]
    ) -> torch.FloatTensor:
        # NOTE: This is an example placeholder, the data source can be easily expanded to include more sentences
        # or be externalized in a public hugging face dataset.
        system_messages_penalizing_sentences = [
            r"here(?:\s+is|\s*'s)\s+a\s+task", # here is a task, here's a task
            r"here(?:\s+is|\s*'s)\s+the\s+solution", # here is the solution, here's the solution
            r"here(?:\s+is|\s*'s)\s+my\s+question", # here is my question, here's my question
            r"what\s+have\s+we\s+learned\s+from\s+this\s+task\?", # what have we learned from this task?
            r"use\s+complete\s+sentences", # use complete sentences
            r"the\s+question\s+was", # the question was
            r"use\s+proper\s+grammar", # use proper grammar
            r"what\s+is\s+the\s+correct\s+order\s+of\s+the\s+key\s+points", # what is the correct order of the key points
            r"sure!\s+here.+", # sure! here...
            r"solution\s+\(in\s+\w+\)", # solution (in \w+)
            r"great\s+job!\s+here(?:'s| is)", # great job! here...
            r"keep\s+it\s+clear\s+and\s+concise.\s+Use\s+complete\s+sentences." # keep it clear and concise. Use complete sentences.
        ]

        penalties = []
        for completion in completions:
            accumulated_penalty = 0.0
            # Trim and consider only the first 200 characters
            completion_segment = completion.strip()[:200].lower()
            for pattern in system_messages_penalizing_sentences:
                if re.search(pattern, completion_segment):
                    accumulated_penalty += 0.1
            penalties.append(accumulated_penalty)

        return torch.tensor(penalties, dtype=torch.float32)
