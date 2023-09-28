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
import unittest
from dataclasses import fields
from unittest.mock import patch
from prompting.validators.event import EventSchema
from prompting.validators.reward import RewardModelType

class EventTestCase(unittest.TestCase):

    def test_event_from_dict_all_forward_columns_match(self):
        """Test that all default columns logged on the forward pass are correctly converted
        """
        # Arrange: Create a dictionary with all columns
        event_dict = {
            'completions': ['test'],
            'completion_times': [0.123],
            'completion_status_messages': ['Success'],
            'completion_status_codes': ['1'],
            'name': 'test-name',
            'block': 1.0,
            'gating_loss': 1.0,
            'uids': [1],
            'prompt': 'test-prompt',
            'step_length': 1.0,
            'best': 'test-best',
            'rewards': [1.0],
            RewardModelType.dahoas.value: [1.0],
            RewardModelType.blacklist.value: [1.0],
            RewardModelType.nsfw.value: [1.0],
            RewardModelType.reciprocate.value: [1.0],
            RewardModelType.diversity.value: [1.0],
            RewardModelType.dpo.value: [1.0],
            RewardModelType.rlhf.value: [1.0],
            RewardModelType.prompt.value: [1.0],
            RewardModelType.relevance.value: [1.0],
            RewardModelType.task_validator.value: [1.0],
            
            RewardModelType.dahoas.value + '_normalized': [1.0],
            RewardModelType.blacklist.value + '_normalized': [1.0],
            RewardModelType.nsfw.value + '_normalized': [1.0],
            RewardModelType.reciprocate.value + '_normalized': [1.0],
            RewardModelType.diversity.value + '_normalized': [1.0],
            RewardModelType.dpo.value + '_normalized': [1.0],
            RewardModelType.rlhf.value + '_normalized': [1.0],
            RewardModelType.prompt.value + '_normalized': [1.0],
            RewardModelType.relevance.value + '_normalized': [1.0],
            RewardModelType.task_validator.value + '_normalized': [1.0]
        }

        # Act
        with patch('bittensor.logging.warning') as mock_warning:
            event = EventSchema.from_dict(event_dict, disable_log_rewards=False)
            mock_warning.assert_not_called()

        # Assert
        for field in fields(EventSchema):
            field_name = field.name
            field_value = getattr(event, field_name)

            # Note: Does not include 'set_weights' column as it is not logged on the forward pass
            if field_name == 'set_weights':
                assert field_value is None
                continue

            print(field_name, field_value)
            assert field_name in event_dict and event_dict[field_name] == field_value


    def test_event_from_dict_forward_no_reward_logging(self):
        """Test that all default columns (not including reward columns) logged on the forward pass are
        correctly converted"""
        # Assert: create a dictionary with all non-related reward columns
        event_dict = {
            'completions': ['test'],
            'completion_times': [0.123],
            'completion_status_messages': ['Success'],
            'completion_status_codes': ['1'],
            'name': 'test-name',
            'block': 1.0,
            'gating_loss': 1.0,
            'uids': [1],
            'prompt': 'test-prompt',
            'step_length': 1.0,
            'best': 'test-best',
            'rewards': [1.0],
        }

        # Act
        with patch('bittensor.logging.warning') as mock_warning:
            event = EventSchema.from_dict(event_dict, disable_log_rewards=True)
            mock_warning.assert_not_called()

        # Assert: Check that all columns that were logged are correctly converted
        for key, value in event_dict.items():
            assert getattr(event, key) == value

        # Assert: Check that all reward columns that are not logged are set to None
        assert event.set_weights is None
        assert event.dahoas_reward_model is None
        assert event.blacklist_filter is None
        assert event.nsfw_filter is None
        assert event.reciprocate_reward_model is None
        assert event.diversity_reward_model is None
        assert event.dpo_reward_model is None
        assert event.rlhf_reward_model is None
        assert event.prompt_reward_model is None
        assert event.relevance_filter is None
        assert event.task_validator_filter is None

        assert event.dahoas_reward_model_normalized is None
        assert event.nsfw_filter_normalized is None
        assert event.reciprocate_reward_model_normalized is None
        assert event.diversity_reward_model_normalized is None
        assert event.dpo_reward_model_normalized is None
        assert event.rlhf_reward_model_normalized is None
        assert event.prompt_reward_model_normalized is None
        assert event.relevance_filter_normalized is None
        assert event.task_validator_filter_normalized is None

    def test_event_from_dict_forward_reward_logging_mismatch(self):
        """Test that all default columns logged on the forward pass are correctly converted and that
        that reward columns that should be logged are logged as warnings"""
        # Assert: create a dictionary with all non-related reward columns
        event_dict = {
            'completions': ['test'],
            'completion_times': [0.123],
            'completion_status_messages': ['Success'],
            'completion_status_codes': ['1'],
            'name': 'test-name',
            'block': 1.0,
            'gating_loss': 1.0,
            'uids': [1],
            'prompt': 'test-prompt',
            'step_length': 1.0,
            'best': 'test-best',
            'rewards': [1.0],
        }

        not_logged_columns = []
        for field in RewardModelType: 
            not_logged_columns.append(field.value)
            if field.value != 'blacklist_filter':
                not_logged_columns.append(field.value + '_normalized')
                

        # Act
        with patch('bittensor.logging.warning') as mock_warning:
            event = EventSchema.from_dict(event_dict, disable_log_rewards=False)
            # Assert: Check that all columns that are not logged in the dict are logged as warnings
            self.assertEqual(mock_warning.call_count, len(not_logged_columns))

        # Assert: Check that all columns that were logged are correctly converted
        for key, value in event_dict.items():
            assert getattr(event, key) == value

        # Assert: Check that all reward columns that are not logged are set to None
        assert event.set_weights is None
        assert event.dahoas_reward_model is None
        assert event.blacklist_filter is None
        assert event.nsfw_filter is None
        assert event.reciprocate_reward_model is None
        assert event.diversity_reward_model is None
        assert event.dpo_reward_model is None
        assert event.rlhf_reward_model is None
        assert event.prompt_reward_model is None
        assert event.relevance_filter is None
        assert event.task_validator_filter is None

        assert event.dahoas_reward_model_normalized is None
        assert event.nsfw_filter_normalized is None
        assert event.reciprocate_reward_model_normalized is None
        assert event.diversity_reward_model_normalized is None
        assert event.dpo_reward_model_normalized is None
        assert event.rlhf_reward_model_normalized is None
        assert event.prompt_reward_model_normalized is None
        assert event.relevance_filter_normalized is None
        assert event.task_validator_filter_normalized is None
