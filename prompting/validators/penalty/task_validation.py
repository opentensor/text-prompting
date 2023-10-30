import torch
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType


class TaskValidationPenaltyModel(BasePenaltyModel):
    @property
    def name(self) -> str:
        return PenaltyModelType.task_validation_penalty.value

    def calculate_penalties(
        self, task: Task, completions: List[str]
    ) -> torch.FloatTensor:
        accumulated_penalties: torch.FloatTensor = torch.zeros(
            len(completions), dtype=torch.float32
        )

        # Accumulate penalties for each criterion
        for criterion in task.criteria:
            accumulated_penalties.add_(criterion.evaluate(completions))

        return accumulated_penalties
