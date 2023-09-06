# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

import wandb
import bittensor as bt
from typing import List, Dict, Union, Tuple, Callable
import time


def record_request_timestamps(self, forward_call: "bt.TextPromptingForwardCall"):
    timestamp_length = self.config.miner.priority.len_request_timestamps
    if forward_call.src_hotkey not in self.request_timestamps:
        self.request_timestamps[forward_call.src_hotkey] = [0] * timestamp_length

    self.request_timestamps[forward_call.src_hotkey].append(forward_call.start_time)
    self.request_timestamps[forward_call.src_hotkey] = self.request_timestamps[
        forward_call.src_hotkey
    ][-timestamp_length:]

    return self.request_timestamps


def default_priority(self, forward_call: "bt.TextPromptingForwardCall") -> float:
    # Check if the key is registered.
    registered = False
    if self.metagraph is not None:
        registered = forward_call.src_hotkey in self.metagraph.hotkeys

    # Non-registered users have a default priority.
    if not registered:
        return self.config.miner.priority.default

    # If the user is registered, it has a UID.
    uid = self.metagraph.hotkeys.index(forward_call.src_hotkey)
    stake_amount = self.metagraph.S[uid].item()

    # request period
    if forward_call.src_hotkey in self.request_timestamps:
        period = time.time() - self.request_timestamps[forward_call.src_hotkey][-10]
        period_scale = period / (
            self.config.miner.priority.time_stake_multiplicate * 60
        )
        priority = max(period_scale, 1) * stake_amount

    else:
        priority = self.config.miner.priority.default

    record_request_timestamps(self, forward_call)

    return priority


def priority(
    self, func: Callable, forward_call: "bt.TextPromptingForwardCall"
) -> float:
    # Check to see if the subclass has implemented a priority function.
    priority = None
    try:
        # Call the subclass priority function and return the result.
        priority = func(forward_call)

    except NotImplementedError:
        # If the subclass has not implemented a priority function, we use the default priority.
        priority = default_priority(self, forward_call)

    except Exception as e:
        # An error occured in the subclass priority function.
        bt.logging.error(f"Error in priority function: {e}")

    finally:
        # If the priority is None, we use the default priority.
        if priority == None:
            priority = default_priority(self, forward_call)

        # Return the priority.
        return priority
