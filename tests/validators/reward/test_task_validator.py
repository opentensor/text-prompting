import unittest
from prompting.validators.reward.task_validator import TaskValidator

class TaskValidatorTestCase(unittest.TestCase):
    """
    This class contains unit tests for the TaskValidator class.

    The tests cover different scenarios for the `reward` method of the TaskValidator class. 
    The `reward` method is expected to return a reward based on the task name and the completion text.
    """
     
    def setUp(self):
        self.validator = TaskValidator()

    def test_augment_with_answer_keyword(self):
        """
        Test if the reward method returns 0 when the task "name" starts with 'augment' (summarization)
        and the completion contains the 'Answer:' keyword.
        """
        name = f'augment'
        completion = "Summary: test summary\nAnswer: Test answer"
        self.assertEqual(self.validator.reward('', completion, name), 0.0)

    def test_followup_with_answer_keyword(self):
        """
        Test if the reward method returns 0 when the task "name" starts with 'followup' (question generation)
        and the completion contains the 'Answer:' keyword.
        """
        for i in range(0, 4):
            name = f'followup{i}'
            completion = 'Question: This is a test question?\nAnswer: This is a test answer.'
            self.assertEqual(self.validator.reward('', completion, name), 0.0)

    def test_augment_with_question_keyword(self):
        """
        Test if the reward method returns 0 when the task "name" starts with 'augment' (summarization)
        and the completion contains the 'Question:' keyword.
        """
        name = f'augment'
        completion = "Summary: test summary\nQuestion: This is a test question?"
        self.assertEqual(self.validator.reward('', completion, name), 0.0)

    def test_answer_with_question_keyword(self):
        """
        Test if the reward method returns 0 when the task "name" is 'answer' (answer generation)
        and the completion contains the 'Question:' keyword.
        """
        for i in range(0, 4):
            name = f'answer{i}'
            completion = 'Question: This is a test question?\nAnswer: This is a test answer.'
            self.assertEqual(self.validator.reward('', completion, name), 0.0)

    def test_followup_and_answer_with_summary_keyword(self):
        """
        Test if the reward method returns 0 when the task "name" is different from "augment" (summarization)
        and the completion contains the 'Summary:' keyword.
        """
        for name in ['followup0', 'followup1', 'followup2', 'followup3', 'answer0', 'answer1', 'answer2', 'answer3']:
            completion = 'Summary: This is a test summary.'
            self.assertEqual(self.validator.reward('', completion, name), 0.0)

    def test_reward_valid_followup(self):
        """
        Test if the reward method returns 1 when the task "name" starts with 'followup' (question generation)
        and the completion contains a question
        """
        for i in range(0, 4):
            name = f'followup{i}'
            completion = 'Question: This is a test question?'
            self.assertEqual(self.validator.reward('', completion, name), 1.0)

    def test_reward_valid_answer(self):
        """
        Test if the reward method returns 1 when the task "name" is 'answer' (answer generation)
        and the completion contains an answer
        """
        for i in range(0, 4):
            name = f'answer{i}'
            completion = 'Answer: This is a test answer.'
            self.assertEqual(self.validator.reward('', completion, name), 1.0)

    def test_reward_valid_augment(self):
        """
        Test if the reward method returns 1 when the task "name" is 'augment' (summarization)
        and the completion contains the a summary.
        """
        name = 'augment'
        completion = 'Summary: This is a test summary.'
        self.assertEqual(self.validator.reward('', completion, name), 1.0)

    def test_reward_valid_other(self):
        """
        Test if the reward method returns 1 when the task "name" is different from "augment", "followup", and "answer"
        and the completion does not contain the 'Summary:', 'Answer:', and 'Question:' keywords.
        """
        for name in ['followup0', 'followup1', 'followup2', 'followup3', 'answer0', 'answer1', 'answer2', 'answer3']:
            completion = 'This is a test completion.'
            self.assertEqual(self.validator.reward('', completion, name), 1.0)

if __name__ == '__main__':
    unittest.main()