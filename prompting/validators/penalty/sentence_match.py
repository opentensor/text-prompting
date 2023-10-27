import torch
from typing import List
from prompting.validators.tasks import Task
from prompting.validators.penalty.penalty import BasePenaltyModel, PenaltyModelType

class SentenceMatchPenaltyModel(BasePenaltyModel):
    def __init__(self, max_penalty: float):
        self.max_penalty = max_penalty
        super().__init__()

    @property
    def name(self) -> str:
        return PenaltyModelType.sentence_length_penalty.value

    @property
    def max_penalty(self) -> float:
        return self.max_penalty
    
    def calculate_penalties(self, task: Task, completions: List[str]) -> torch.FloatTensor:
        # NOTE: This is an example placeholder, the data source can be easily expanded to include more sentences 
        # or be externalized in a public hugging face dataset.
        penalizing_sentences = [
            "what have we learned from this task?",
            "here is a task",
            "here is the solution",
            "use complete sentences",
            #...
        ]
        
        penalties = []
        for completion in completions:
            accumulated_penalty = 0.0
            for sentence in penalizing_sentences:
                if sentence in completion.lower():
                    accumulated_penalty += 0.1
                    
            final_penalty = 1 - accumulated_penalty if accumulated_penalty < 1 else 0
            penalties.append(final_penalty)
        
        return torch.tensor(penalties, dtype=torch.float32)
