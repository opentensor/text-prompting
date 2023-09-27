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

import time
import wandb
import random
import bittensor as bt
import traceback
from typing import Callable, Dict, List, Union

from .blacklist import is_prompt_in_cache


def forward(
    self,
    func: Callable,
    messages: List[Dict[str, str]],
    log_data: Dict[str, Union[str, float]] = None,
) -> str:
    """Forwards a list of messages to the miner's forward function."""

    # Check prompt cache to see if we should blacklist.
    if is_prompt_in_cache(self, messages):
        raise Exception(f"Prompt in cache, rejecting. {messages}")

    # Run the subclass forward function.
    try:
        start_time = time.time()
        response = func(messages)
        success = 1

    # There was an error in the error function.
    except Exception as e:
        bt.logging.error(f"Error in forward function: { e }")
        response = ""
        success = 0

    finally:
        # Log the response length and qtime.
        if self.config.wandb.on:
            wandb_log_data = {
                "messages": messages,
                "completion": response,
                "end_time": time.time(),
                "block": self.subtensor.block,
                "forward_elapsed": time.time() - start_time,
                "forward_success": success,
            }

            wandb.log(
                wandb_log_data if log_data == None else {**log_data, **wandb_log_data}
            )

        # Return the response.
        return response
