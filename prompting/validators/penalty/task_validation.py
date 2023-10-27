import torch
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

class TaskValidationPenaltyModel(BasePenaltyModel):    
    def __init__(self, max_penalty: float):
        self.max_penalty = max_penalty
        super().__init__()

    @property
    def name(self) -> str:
        return PenaltyModelType.task_validation_penalty.value

    @property
    def max_penalty(self) -> float:
        return self.max_penalty
    
    def calculate_penalties(self, task: Task, completions: List[str]) -> torch.FloatTensor:
        accumulated_penalties: torch.FloatTensor = torch.zeros(len(completions), dtype=torch.float32)
        
        # Accumulate penalties for each criterion        
        for criterion in task.criteria:            
            accumulated_penalties.add_(criterion.evaluate(completions))

        # Update accumulated_penalties based on conditions
        mask_less_than_1 = accumulated_penalties < 1
        mask_greater_or_equal_1 = accumulated_penalties >= 1
        accumulated_penalties[mask_less_than_1] = 1 - accumulated_penalties[mask_less_than_1]
        accumulated_penalties[mask_greater_or_equal_1] = 0
                     
        return accumulated_penalties