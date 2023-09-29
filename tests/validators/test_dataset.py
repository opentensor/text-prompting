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
from prompting.validators.dataset import Dataset


class DatasetTestCase(unittest.TestCase):
    def test_next_skips_empty_and_newline_only_strings(self):
        mock_data = iter([{"text": ""}, {"text": "\n\n"}, {"text": "Non-empty text"}])
        dataset = Dataset()
        dataset.openwebtext = mock_data
        dataset.red_pajama = mock_data

        # Test that __next__ skips empty texts and texts that consist only of newline characters
        self.assertEqual(dataset.__next__(), {"text": "Non-empty text"})

    def test_next_returns_regular_strings(self):
        mock_data = iter([{"text": "Non-empty text"}])
        dataset = Dataset()
        dataset.openwebtext = mock_data
        dataset.red_pajama = mock_data

        # Test that __next__ returns a non-empty text
        self.assertEqual(dataset.__next__(), {"text": "Non-empty text"})


if __name__ == "__main__":
    unittest.main()
