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
# import torch
# import copy
# import asyncio
# import sys
# from neurons.validators.validator import neuron as Neuron
# from prompting.validators.forward import run_step
# from unittest.mock import MagicMock, patch

# from .helpers import __mock_wallet_factory__

# CLI_ARGS_STR = "validators/openvalidators/neuron.py --mock --wallet._mock --wandb.off --neuron.followup_sample_size 5 --neuron.answer_sample_size 5"

# SYS_ARGV = sys.argv.copy()


# patcher = None

# def setUpModule():
#     """Runs once for the tests in this module."""
#     global patcher
#     patcher = patch("bittensor.wallet.__new__", __mock_wallet_factory__ )
#     patcher.start()

# def tearDownModule():
#     """Runs once for the tests in this module."""
#     global patcher
#     if patcher:
#         patcher.stop()


# def test_uid_weights_unchanged_unless_queried(n_steps=10, n_concurrent=1):
#     """Test that the weights of unqueried uids do not over the course of a forward pass."""

#     sys.argv = CLI_ARGS_STR.split(" ")
#     neuron = Neuron()
#     neuron.blacklist = MagicMock(return_value =True)

#     for _ in range(n_steps):

#         prev_scores = copy.deepcopy(neuron.moving_averaged_scores)

#         async def forward(neuron):
#             # Request a run step
#             event = await run_step( 
#                 neuron, 
#                 prompt = '', 
#                 name = 'augment',
#                 k = 10,
#                 timeout = 10,
#             )
#             return event

#         # run concurrent forward passes
#         async def run_forward():
#             coroutines = [forward(neuron) for _ in range(n_concurrent)]
#             return await asyncio.gather(*coroutines)

#         events = neuron.loop.run_until_complete(run_forward())
#         # moving_averaged_scores updates are not thread safe, so I don't think we can run concurrent forwards
#         for event in events:

#             # get current scores
#             next_scores = copy.deepcopy(neuron.moving_averaged_scores)

#             queried_uids = sorted(set(event["uids"]))
#             ignored_uids = [uid for uid in torch.arange(neuron.metagraph.n.item()) if uid not in queried_uids]

#             # ther is a floating point difference (~1e-10) between the scores, so we can't use exact equality
#             assert next_scores[ignored_uids].allclose(prev_scores[ignored_uids]), "Unqueried uids should not change"

#     sys.argv = SYS_ARGV.copy()
