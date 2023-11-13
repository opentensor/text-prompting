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
# DEALINGS IN
#  THE SOFTWARE.

import time
import torch
import random
import bittensor as bt
import random

from loguru import logger
from typing import List
from dataclasses import asdict
from prompting.validators.event import EventSchema
from prompting.validators.misc import ttl_get_block
from prompting.validators.prompts import followup_prompt, answer_prompt, augment_prompt
from prompting.validators.utils import check_uid_availability
from prompting.validators.tasks import (
    Task,
    create_summarization_task,
    create_qg_task,
    create_qa_task,
)

import prompting


def get_random_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = torch.tensor(random.sample(available_uids, k))
    return uids


async def run_step(self, task: Task, k: int, timeout: float, exclude: list = []):
    task_name = task.task_name
    prompt = task.compose_prompt()

    bt.logging.debug("run_step", task_name)

    # Record event start time.
    event = {"name": task_name, "task_type": task.task_type}
    start_time = time.time()
    # Get the list of uids to query for this step.
    uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)
    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = prompting.protocol.Prompting(roles=["user"], messages=[prompt])

    # Make calls to the network with the prompt.
    responses: List[bt.Synapse] = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )

    # Update blacklist with completions so that n-gram filtering can be applied
    self.blacklist.add(
        [response.completion for response in responses if response.completion]
    )

    # Restrict the format of acceptable followup completions.
    for response in responses:
        # remove leading and trailing periods
        completion = response.completion.strip(".")

        if "followup" in task_name and len(completion) > 0:
            # take maximum of 40 words
            max_words = 40
            if "?" in completion:
                # take first question that is found and only use the sentence before the question mark
                completion = completion.split("?")[0].split(".")[-1]
                response.completion = " ".join(completion.split(" ")[-max_words:]) + "?"
            else:
                # otherwise take the last sentence
                completion = completion.split(".")[-1].split(".")[-1]
                response.completion = " ".join(completion.split(" ")[-max_words:])

    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = torch.zeros(len(responses), dtype=torch.float32).to(
        self.device
    )
    for weight_i, reward_fn_i in zip(self.reward_weights, self.reward_functions):
        reward_i_normalized, reward_event = reward_fn_i.apply(
            task.base_text, responses, task_name
        )
        rewards += weight_i * reward_i_normalized.to(self.device)
        if not self.config.neuron.disable_log_rewards:
            event = {**event, **reward_event}
        bt.logging.trace(str(reward_fn_i.name), reward_i_normalized.tolist())

    for masking_fn_i in self.masking_functions:
        mask_i_normalized, reward_event = masking_fn_i.apply(
            task.base_text, responses, task_name
        )
        rewards *= mask_i_normalized.to(self.device)  # includes diversity
        if not self.config.neuron.disable_log_rewards:
            event = {**event, **reward_event}
        bt.logging.trace(str(masking_fn_i.name), mask_i_normalized.tolist())

    for penalty_fn_i in self.penalty_functions:
        (
            raw_penalty_i,
            adjusted_penalty_i,
            applied_penalty_i,
        ) = penalty_fn_i.apply_penalties(responses, task)
        rewards *= applied_penalty_i.to(self.device)
        if not self.config.neuron.disable_log_rewards:
            event[penalty_fn_i.name + "_raw"] = raw_penalty_i.tolist()
            event[penalty_fn_i.name + "_adjusted"] = adjusted_penalty_i.tolist()
            event[penalty_fn_i.name + "_applied"] = applied_penalty_i.tolist()
        bt.logging.trace(str(penalty_fn_i.name), applied_penalty_i.tolist())

    # Train the gating model based on the predicted scores and the actual rewards.
    gating_scores: torch.FloatTensor = self.gating_model(prompt).to(self.device)
    gating_loss: torch.FloatTensor = self.gating_model.backward(
        scores=gating_scores[uids], rewards=rewards
    )

    # Find the best completion given the rewards vector.
    completions: List[str] = [comp.completion for comp in responses]
    completion_status_message: List[str] = [
        str(comp.dendrite.status_message) for comp in responses
    ]
    completion_status_codes: List[str] = [
        str(comp.dendrite.status_code) for comp in responses
    ]

    best: str = completions[rewards.argmax(dim=0)].strip()

    # Get completion times
    completion_times: List[float] = [
        comp.dendrite.process_time if comp.dendrite.process_time != None else 0
        for comp in responses
    ]

    # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # shape: [ metagraph.n ]
    scattered_rewards: torch.FloatTensor = self.moving_averaged_scores.scatter(
        0, uids, rewards
    ).to(self.device)

    # Update moving_averaged_scores with rewards produced by this step.
    # shape: [ metagraph.n ]
    alpha: float = self.config.neuron.moving_average_alpha
    self.moving_averaged_scores: torch.FloatTensor = alpha * scattered_rewards + (
        1 - alpha
    ) * self.moving_averaged_scores.to(self.device)

    # Log the step event.
    event.update(
        {
            "block": ttl_get_block(self),
            "step_length": time.time() - start_time,
            "prompt": prompt,
            "uids": uids.tolist(),
            "completions": completions,
            "completion_times": completion_times,
            "completion_status_messages": completion_status_message,
            "completion_status_codes": completion_status_codes,
            "rewards": rewards.tolist(),
            "gating_loss": gating_loss.item(),
            "best": best,
        }
    )

    bt.logging.debug("event:", str(event))
    if not self.config.neuron.dont_save_events:
        logger.log("EVENTS", "events", **event)

    # Log the event to wandb.
    if not self.config.wandb.off:
        wandb_event = EventSchema.from_dict(
            event, self.config.neuron.disable_log_rewards
        )
        self.wandb.log(asdict(wandb_event))

    # Return the event.
    return event


async def forward(self):
    # Obtain a unique context from the dataset.
    data = next(self.dataset)["text"]

    random_cutoff = random.randint(15, 30)
    # Truncate context to a limited set of sentences.
    base_text = ".".join(data.split(".", maxsplit=random_cutoff)[:-1])

    # Create a summary task from the context.
    summary_task: Task = create_summarization_task(base_text)    

    # Request a summary, given the original context.
    summarization_event = await run_step(
        self,
        task=summary_task,
        k=self.config.neuron.followup_sample_size,
        timeout=self.config.neuron.followup_timeout,
    )

    best_summary = summarization_event["best"]
    exclude = summarization_event["uids"]
    best_summary_context = "### SUMMARY CONTEXT:\n" + best_summary

    for k in range(self.config.neuron.num_followup_steps):
        # Get a followup question, given the summarized context.
        qg_task = create_qg_task(base_text=best_summary_context, index=k)
        qg_event = await run_step(
            self,
            task=qg_task,
            k=self.config.neuron.followup_sample_size,
            timeout=self.config.neuron.followup_timeout,
            exclude=exclude,
        )
        exclude += qg_event["uids"]

        # Adds the best question to the prompt context.
        best_question = qg_event["best"]
        best_question_prompt = (
            best_summary_context + f"\n### QUESTION {k}:\n{best_question}"
        )

        qa_task = create_qa_task(best_question_prompt, index=k)
        qa_event = await run_step(
            self,
            task=qa_task,
            k=self.config.neuron.answer_sample_size,
            timeout=self.config.neuron.answer_timeout,
            exclude=exclude,
        )

        exclude += qa_event["uids"]
