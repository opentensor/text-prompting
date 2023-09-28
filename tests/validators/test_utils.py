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
import torch
import bittensor as bt
import copy
import unittest
from unittest.mock import MagicMock
from prompting.validators.utils import resync_linear_layer, check_uid_availability


class UtilsTestCase(unittest.TestCase):
    def setUp(self):
        """
        Creates a mock metagraph with 1024 mock axons before each test.
        """
        mock_metagraph = MagicMock(spec=bt.metagraph)
        mock_metagraph.uids = torch.tensor(range(0, 1024))
        mock_metagraph.S = torch.zeros(1024)
        mock_metagraph.hotkeys = list(map(str, range(0, 1024)))
        mock_metagraph.validator_permit = [False] * 1024
        mock_metagraph.axons = [
            MagicMock(spec=bt.chain_data.AxonInfo, hotkey=str(num), ip="0.0.0.0/0", port=12345) for num in range(0, 1024)
        ]

        self.metagraph = mock_metagraph
        self.keypair = "test"

    def test_resync_linear_layer_multiple_updates(self):
        # Arrange: Create necessary inputs for the test
        # Create a linear layer of 768 x uids full of ones
        linear_output_size = len(self.metagraph.uids)
        linear_layer = torch.nn.Linear(768, linear_output_size)
        torch.nn.init.ones_(linear_layer.weight)
        torch.nn.init.ones_(linear_layer.bias)

        # Create a new metagraph state with updated hotkeys
        updated_uids_indices = [0, 10, 20, 30]
        modified_metagraph = copy.deepcopy(self.metagraph)

        for modified_index in updated_uids_indices:
            modified_metagraph.hotkeys[modified_index] = "test"

        # Act: Call the utils function to be tested
        resync_linear_layer(linear_layer, self.metagraph, modified_metagraph)

        # Assert: Ensure that the bias of the updated indices have been reinitialized as expected
        for index in range(0, linear_output_size):
            # If the index has been updated, assert that bias is zero and weights are not ones
            if index in updated_uids_indices:
                self.assertEqual(linear_layer.bias[index].item(), 0)
                self.assertFalse(torch.all(linear_layer.weight[index] == torch.ones(linear_layer.weight[index].shape)))
            # If the index has not been updated, assert that bias is one and weights are ones
            else:
                self.assertEqual(linear_layer.bias[index].item(), 1)
                self.assertTrue(torch.all(linear_layer.weight[index] == torch.ones(linear_layer.weight[index].shape)))

    def test_check_uid_availability_not_serving_axon(self):
        # Arrange: Create a non serving axon
        uid = 1
        self.metagraph.axons[uid] = MagicMock(spec=bt.chain_data.AxonInfo, is_serving=False)

        # Act: Call the function to check if uid is available
        result = check_uid_availability(self.metagraph, uid, vpermit_tao_limit=0)

        # Assert: Ensure that the result is False (uid is available) when node doesn't have a serving axon
        self.assertFalse(result)

    def test_check_uid_availability_node_without_validator_permit(self):
        # Arrange: Create a serving axon without validator permit
        uid = 1
        self.metagraph.axons[uid] = MagicMock(spec=bt.chain_data.AxonInfo, is_serving=True)
        self.metagraph.validator_permit[uid] = False

        # Act: Call the function to check if uid is available
        result = check_uid_availability(self.metagraph, uid, vpermit_tao_limit=0)

        # Assert: Ensure that the result is True (uid is available) when node does not have a validator permit
        self.assertTrue(result)

    def test_check_uid_availability_validator_with_stake_less_than_vpermit_tao_limit(self):
        # Arrange: Create a serving axon with validator permit and stake less than vpermit_tao_limit
        uid = 1
        self.metagraph.axons[uid] = MagicMock(spec=bt.chain_data.AxonInfo, is_serving=True)
        self.metagraph.validator_permit[uid] = True
        self.metagraph.S[uid] = 1
        v_permit_tao_limit = 2

        # Act: Call the function to check if uid is available
        result = check_uid_availability(self.metagraph, uid, vpermit_tao_limit=v_permit_tao_limit)

        # Assert: Ensure that the result is True (uid is available) when node validator
        # has stake less than vpermit_tao_limit
        self.assertTrue(result)

    def test_check_uid_availability_validator_with_stake_greater_than_vpermit_tao_limit(self):
        # Arrange: Create a serving axon with validator permit and stake greater than vpermit_tao_limit
        uid = 1
        self.metagraph.axons[uid] = MagicMock(spec=bt.chain_data.AxonInfo, is_serving=True)
        self.metagraph.validator_permit[uid] = True
        self.metagraph.S[uid] = 2
        v_permit_tao_limit = 1

        # Act: Call the function to check if uid is available
        result = check_uid_availability(self.metagraph, uid, vpermit_tao_limit=v_permit_tao_limit)

        # Assert: Ensure that the result is False (uid is available) when validator node
        # has stake greater than vpermit_tao_limit
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
