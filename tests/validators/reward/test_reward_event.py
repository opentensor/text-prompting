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

import unittest
from dataclasses import fields
import prompting.validators.reward as reward


class RewardEventTestCase(unittest.TestCase):
    """
    This class contains unit tests for the RewardEvent classes.

    The tests cover different scenarios where completions may or may not be successful and the reward events are checked that they don't contain missing values.
    The `reward` attribute of all RewardEvents is expected to be a float, and the `is_filter_model` attribute is expected to be a boolean.
    """

    def setUp(self):
        self.event_classes = [
            reward.reward.BaseRewardEvent,  # Represents a reward model (float)
            reward.nsfw.NSFWRewardEvent,  # Remaining events are filters
            reward.blacklist.BlacklistRewardEvent,
            reward.relevance.RelevanceRewardEvent,
            reward.diversity.DiversityRewardEvent,
        ]

        self.reward_events = {}
        for event in self.event_classes:
            event_type = event.__name__
            self.reward_events[event_type] = []

            # Simulate a batch of completions
            for i in range(50):
                ev = event()

                # Simulate unsuccessful completions by leaving reward event as its default value
                if i % 10 == 0:
                    continue

                for field in fields(ev):
                    # don't modify the is_filter_model field
                    if field.name == "is_filter_model":
                        continue
                    # otherwise set the field to a float (including reward)
                    setattr(ev, field.name, 1.234)

                self.reward_events[event_type].append(ev)

    def test_no_missing_rewards(self):
        for name, events in self.reward_events.items():
            parsed = reward.reward.BaseRewardEvent.parse_reward_events(events)

            # Ensure that all rewards are not None
            self.assertTrue(
                all(r is not None for r in parsed["reward"]),
                f"Events for {name} are missing rewards",
            )

    def test_imputed_reward_values_are_correct(self):
        for name, events in self.reward_events.items():
            expected_value = 1
            indices_missing_reward = [
                i for i, ev in enumerate(events) if ev.reward is None
            ]

            parsed = reward.reward.BaseRewardEvent.parse_reward_events(events)

            # Ensure that all rewards are not None
            self.assertTrue(
                all(
                    parsed["reward"][i] == expected_value
                    for i in indices_missing_reward
                ),
                f"Events for {name} were imputed with incorrect reward value",
            )

    def test_parse_reward_events_with_reward_events(self):
        # Create sample reward events
        event1 = reward.reward.BaseRewardEvent(reward=1, normalized_reward="event1")
        event2 = reward.reward.BaseRewardEvent(reward=2, normalized_reward="event2")
        events = [event1, event2]

        # Expected result
        expected = {"reward": (1, 2), "normalized_reward": ("event1", "event2")}

        # Call the function and check if the result matches the expected output
        result = reward.reward.BaseRewardEvent.parse_reward_events(events)
        self.assertEqual(result, expected)

    def test_parse_reward_events_with_no_reward_events(self):
        # Test with None
        result_none = reward.reward.BaseRewardEvent.parse_reward_events(None)
        self.assertTrue(all(len(lst) == 0 for lst in result_none.values()))
        self.assertEqual(result_none, {"reward": [], "normalized_reward": []})

        # Test with empty list
        result_empty = reward.reward.BaseRewardEvent.parse_reward_events([])
        self.assertTrue(all(len(lst) == 0 for lst in result_empty.values()))
        self.assertEqual(result_empty, {"reward": [], "normalized_reward": []})
