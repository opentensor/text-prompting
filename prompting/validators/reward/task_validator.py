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
from typing import List, Union
from .config import RewardModelType
from .reward import BaseRewardModel, BaseRewardEvent


class TaskValidator(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.task_validator.value

    def __init__(self):
        super().__init__()

    def reward(self, prompt: str, completion: str, name: str) -> BaseRewardEvent:
        reward_event = BaseRewardEvent()

        summary_keywords = ["Summary:", "Paraphrase:", "Paraphrasing:", "Paraphrased:"]
        question_keywords = ["Question:", "Query:", "Q:"]
        answer_keywords = ["Answer:", "Response:", "A:", "Completion:"]

        completion_contains_answer = any(
            answer_keyword.lower() in completion.lower()
            for answer_keyword in answer_keywords
        )
        completion_contains_question = any(
            question_keyword.lower() in completion.lower()
            for question_keyword in question_keywords
        )
        completion_contains_summary = any(
            summary_keyword.lower() in completion.lower()
            for summary_keyword in summary_keywords
        )

        is_summarization_prompt = name == "augment"
        is_question_prompt = name.startswith("followup")
        is_answer_prompt = name.startswith("answer")

        if (
            is_summarization_prompt or is_question_prompt
        ) and completion_contains_answer:
            reward_event.reward = 0.0
            return reward_event

        if (
            is_summarization_prompt or is_answer_prompt
        ) and completion_contains_question:
            reward_event.reward = 0.0
            return reward_event

        if not is_summarization_prompt and completion_contains_summary:
            reward_event.reward = 0.0
            return reward_event

        reward_event.reward = 1
        return reward_event

    def get_rewards(self, prompt: str, completions: List[str], name: str) -> List[BaseRewardEvent]:
        # Get all the reward results.
        reward_events = [
            self.reward(prompt, completion, name) for completion in completions
        ]

        return reward_events

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

    def reset(self):
        pass
