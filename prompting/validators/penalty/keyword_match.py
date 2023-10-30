import torch
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType


class KeywordMatchPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.keyword_match_penalty.value

    def check_exploits_keywords(self, completion: str, name: str) -> float:
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
            return 1

        if (
            is_summarization_prompt or is_answer_prompt
        ) and completion_contains_question:
            return 1

        if not is_summarization_prompt and completion_contains_summary:
            return 1

        return 0

    def calculate_penalties(
        self, task: Task, completions: List[str]
    ) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.check_exploits_keywords(completion, task.task_name)
                for completion in completions
            ],
            dtype=torch.float32,
        )
