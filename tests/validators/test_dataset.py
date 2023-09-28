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


if __name__ == '__main__':
    unittest.main()