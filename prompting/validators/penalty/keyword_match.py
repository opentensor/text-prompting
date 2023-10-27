import torch
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

class KeywordMatchPenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float):
        self.max_penalty = max_penalty
        super().__init__()

    @property
    def name(self) -> str:
        return PenaltyModelType.keyword_match_penalty.value

    @property
    def max_penalty(self) -> float:
        return self.max_penalty
    
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
            return 0.0

        if (
            is_summarization_prompt or is_answer_prompt
        ) and completion_contains_question:
            return 0.0

        if not is_summarization_prompt and completion_contains_summary:
            return 0.0

        return 1


    def calculate_penalties(self, task: Task, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor(
            [self.reward(completion, task.task_name) for completion in completions],
            dtype=torch.float32,
        )
