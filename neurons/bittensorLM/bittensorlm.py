# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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


import os
import time
import argparse
import bittensor as bt
import deepspeed
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

from baseminer.revolution_miner import Miner
from prompting.protocol import Prompting


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class CerebrasBTLMMiner(Miner):
    @classmethod
    def config(cls) -> "bt.Config":
        parser = argparse.ArgumentParser(description="Bittensor-LM Miner Config")
        cls.add_args(parser)
        return bt.config(parser)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--btlm.device", type=str, help="Device to load model", default="cuda"
        )
        parser.add_argument(
            "--btlm.max_length",
            type=int,
            default=100,
            help="The maximum length (in tokens) of the generated text.",
        )
        parser.add_argument(
            "--btlm.do_sample",
            action="store_true",
            default=False,
            help="Whether to use sampling or not (if not, uses greedy decoding).",
        )
        parser.add_argument(
            "--btlm.no_repeat_ngram_size",
            type=int,
            default=2,
            help="The size of the n-grams to avoid repeating in the generated text.",
        )
        parser.add_argument(
            "--btlm.do_prompt_injection",
            action="store_true",
            default=False,
            help='Whether to use a custom "system" prompt instead of the one sent by bittensor.',
        )
        parser.add_argument(
            "--btlm.system_prompt",
            type=str,
            help="What prompt to replace the system prompt with",
            default="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. ",
        )
        parser.add_argument(
            "--btlm.use_deepspeed",
            action="store_true",
            default=False,
            help="Whether to use deepspeed or not (if not, uses vanilla huggingface).",
        )
        parser.add_argument(
            "--btlm.temperature", type=float, default=0.7, help="Sampling temperature."
        )

    def __init__(self, *args, **kwargs):
        super(CerebrasBTLMMiner, self).__init__(*args, **kwargs)

        bt.logging.info("Loading BTLM 3B model...")
        model = AutoModelForCausalLM.from_pretrained(
            "cerebras/btlm-3b-8k-base",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "cerebras/btlm-3b-8k-base",
            trust_remote_code=True,
        )
        self.stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
        self.stop = StopOnTokens(self.stop_token_ids)

        # Determine correct device id (int) from device string.
        if self.config.btlm.device == "cuda":
            self.config.btlm.device = 0
        elif len(self.config.btlm.device.split(":") == 2):
            try:
                self.config.btlm.device = int(self.config.btlm.device.split(":")[1])
            except:
                raise ValueError(
                    "Invalid device string: {}".format(self.config.btlm.device)
                )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.config.btlm.device,
            do_sample=self.config.btlm.do_sample,
            max_new_tokens=self.config.btlm.max_length,
            no_repeat_ngram_size=self.config.btlm.no_repeat_ngram_size,
        )

        if self.config.btlm.use_deepspeed:
            self.pipe.model = deepspeed.init_inference(
                self.pipe.model,
                mp_size=int(os.getenv("WORLD_SIZE", "1")),
                dtype=torch.float,
                replace_with_kernel_inject=False,
            )

    def _process_history(self, roles: List[str], messages: List[str]) -> str:
        processed_history = ""
        if self.config.btlm.do_prompt_injection:
            processed_history += self.config.btlm.system_prompt
        for role, message in zip(roles, messages):
            if role == "system":
                if not self.config.btlm.do_prompt_injection or message != history[0]:
                    processed_history += "system: " + message + "\n"
            if role == "assistant":
                processed_history += "assistant: " + message + "\n"
            if role == "user":
                processed_history += "user: " + message + "\n"
        return processed_history

    def prompt(self, synapse: Prompting) -> Prompting:
        history = self._process_history(roles=synapse.roles, messages=synapse.messages)
        history += "assistant: "
        bt.logging.debug("History: {}".format(history))
        completion = (
            self.pipe(
                history,
                temperature=self.config.btlm.temperature,
                max_new_tokens=self.config.btlm.max_length,
                no_repeat_ngram_size=self.config.btlm.no_repeat_ngram_size,
                do_sample=self.config.btlm.do_sample,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([self.stop]),
            )[0]["generated_text"]
            .split(":")[-1]
            .replace(str(history), "")
        )
        bt.logging.debug("Completion: {}".format(completion))
        synapse.completion = completion
        return synapse


if __name__ == "__main__":
    bt.debug()
    miner = CerebrasBTLMMiner()
    with miner:
        while True:
            time.sleep(1)
