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

import bittensor as bt
from dataclasses import dataclass
from typing import List, Optional
from prompting.validators.reward import RewardModelType
from prompting.validators.penalty import PenaltyModelType


@dataclass
class EventSchema:
    completions: List[str]  # List of completions received for a given prompt
    completion_times: List[float]  # List of completion times for a given prompt
    completion_status_messages: List[
        str
    ]  # List of completion status messages for a given prompt
    completion_status_codes: List[
        str
    ]  # List of completion status codes for a given prompt
    name: str  # Prompt type, e.g. 'followup', 'answer'
    task_type: str  # Task type, e.g. 'summary', 'question'
    block: float  # Current block at given step
    gating_loss: float  # Gating model loss for given step
    uids: List[int]  # Queried uids
    prompt: str  # Prompt text string
    step_length: float  # Elapsed time between the beginning of a run step to the end of a run step
    best: str  # Best completion for given prompt

    # Reward data
    rewards: List[float]  # Reward vector for given step
    dahoas_reward_model: Optional[
        List[float]
    ]  # Output vector of the dahoas reward model
    blacklist_filter: Optional[List[float]]  # Output vector of the blacklist filter
    nsfw_filter: Optional[List[float]]  # Output vector of the nsfw filter
    reciprocate_reward_model: Optional[
        List[float]
    ]  # Output vector of the reciprocate reward model
    diversity_reward_model: Optional[
        List[float]
    ]  # Output vector of the diversity reward model
    dpo_reward_model: Optional[List[float]]  # Output vector of the dpo reward model
    rlhf_reward_model: Optional[List[float]]  # Output vector of the rlhf reward model
    prompt_reward_model: Optional[
        List[float]
    ]  # Output vector of the prompt reward model
    relevance_filter: Optional[List[float]]
    dahoas_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the dahoas reward model
    nsfw_filter_normalized: Optional[List[float]]  # Output vector of the nsfw filter
    reciprocate_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the reciprocate reward model
    diversity_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the diversity reward model
    dpo_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the dpo reward model
    rlhf_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the rlhf reward model
    prompt_reward_model_normalized: Optional[
        List[float]
    ]  # Output vector of the prompt reward model
    relevance_filter_normalized: Optional[List[float]]
    # TODO: Add comments
    task_validation_penalty_raw: Optional[List[float]]
    task_validation_penalty_adjusted: Optional[List[float]]
    task_validation_penalty_applied: Optional[List[float]]

    keyword_match_penalty_raw: Optional[List[float]]
    keyword_match_penalty_adjusted: Optional[List[float]]
    keyword_match_penalty_applied: Optional[List[float]]

    sentence_length_penalty_raw: Optional[List[float]]
    sentence_length_penalty_adjusted: Optional[List[float]]
    sentence_length_penalty_applied: Optional[List[float]]

    # Weights data
    set_weights: Optional[List[List[float]]]

    @staticmethod
    def from_dict(event_dict: dict, disable_log_rewards: bool) -> "EventSchema":
        """Converts a dictionary to an EventSchema object."""
        rewards = {
            "blacklist_filter": event_dict.get(RewardModelType.blacklist.value),
            "dahoas_reward_model": event_dict.get(RewardModelType.dahoas.value),
            "nsfw_filter": event_dict.get(RewardModelType.nsfw.value),
            "relevance_filter": event_dict.get(RewardModelType.relevance.value),
            "reciprocate_reward_model": event_dict.get(
                RewardModelType.reciprocate.value
            ),
            "diversity_reward_model": event_dict.get(RewardModelType.diversity.value),
            "dpo_reward_model": event_dict.get(RewardModelType.dpo.value),
            "rlhf_reward_model": event_dict.get(RewardModelType.rlhf.value),
            "prompt_reward_model": event_dict.get(RewardModelType.prompt.value),
            "dahoas_reward_model_normalized": event_dict.get(
                RewardModelType.dahoas.value + "_normalized"
            ),
            "nsfw_filter_normalized": event_dict.get(
                RewardModelType.nsfw.value + "_normalized"
            ),
            "relevance_filter_normalized": event_dict.get(
                RewardModelType.relevance.value + "_normalized"
            ),
            "reciprocate_reward_model_normalized": event_dict.get(
                RewardModelType.reciprocate.value + "_normalized"
            ),
            "diversity_reward_model_normalized": event_dict.get(
                RewardModelType.diversity.value + "_normalized"
            ),
            "dpo_reward_model_normalized": event_dict.get(
                RewardModelType.dpo.value + "_normalized"
            ),
            "rlhf_reward_model_normalized": event_dict.get(
                RewardModelType.rlhf.value + "_normalized"
            ),
            "prompt_reward_model_normalized": event_dict.get(
                RewardModelType.prompt.value + "_normalized"
            ),
        }
        penalties = {
            "task_validation_penalty_raw": event_dict.get(
                PenaltyModelType.task_validation_penalty.value + "_raw"
            ),
            "task_validation_penalty_adjusted": event_dict.get(
                PenaltyModelType.task_validation_penalty.value + "_adjusted"
            ),
            "task_validation_penalty_applied": event_dict.get(
                PenaltyModelType.task_validation_penalty.value + "_applied"
            ),
            "keyword_match_penalty_raw": event_dict.get(
                PenaltyModelType.keyword_match_penalty.value + "_raw"
            ),
            "keyword_match_penalty_adjusted": event_dict.get(
                PenaltyModelType.keyword_match_penalty.value + "_adjusted"
            ),
            "keyword_match_penalty_applied": event_dict.get(
                PenaltyModelType.keyword_match_penalty.value + "_applied"
            ),
            "sentence_length_penalty_raw": event_dict.get(
                PenaltyModelType.sentence_length_penalty.value + "_raw"
            ),
            "sentence_length_penalty_adjusted": event_dict.get(
                PenaltyModelType.sentence_length_penalty.value + "_adjusted"
            ),
            "sentence_length_penalty_applied": event_dict.get(
                PenaltyModelType.sentence_length_penalty.value + "_applied"
            ),
        }

        # Logs warning that expected data was not set properly
        if not disable_log_rewards and any(value is None for value in rewards.values()):
            for key, value in rewards.items():
                if value is None:
                    bt.logging.warning(
                        f"EventSchema.from_dict: {key} is None, data will not be logged"
                    )

        return EventSchema(
            completions=event_dict["completions"],
            completion_times=event_dict["completion_times"],
            completion_status_messages=event_dict["completion_status_messages"],
            completion_status_codes=event_dict["completion_status_codes"],
            name=event_dict["name"],
            task_type=event_dict["task_type"],
            block=event_dict["block"],
            gating_loss=event_dict["gating_loss"],
            uids=event_dict["uids"],
            prompt=event_dict["prompt"],
            step_length=event_dict["step_length"],
            best=event_dict["best"],
            rewards=event_dict["rewards"],
            **rewards,
            **penalties,
            set_weights=None,
        )
