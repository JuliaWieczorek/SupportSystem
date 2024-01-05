import unittest
from flow import divide_and_analyze_conversation
from sklearn.feature_extraction.text import TfidfVectorizer

class TestDivideAndAnalyzeConversation(unittest.TestCase):

    def test_divide_into_three_parts(self):
        dialog_tokens = ["hello", "world", "how", "are", "you", "today", "hello", "world", "how", "are", "you", "today", "hello", "world", "how", "are", "you", "today"]
        expected = [6, 6, 6]
        result = divide_and_analyze_conversation(dialog_tokens, num_parts=3)
        self.assertEqual(expected, result)

    def test_divide_into_two_parts(self):
        dialog_tokens = ["hello", "world", "how", "are", "you", "today", "hello", "world", "how", "are", "you", "today"]
        expected = [6, 6]
        result = divide_and_analyze_conversation(dialog_tokens, num_parts=2)
        self.assertEqual(expected, result)

    def test_empty_dialog(self):
        dialog_tokens = []
        expected = 0
        result = divide_and_analyze_conversation(dialog_tokens)
        self.assertEqual(expected, result)

    def test_one_part_dialog(self):
        dialog_tokens = ["hello", "world", "how", "are", "you", "today"]
        expected = [2, 2, 2]
        result = divide_and_analyze_conversation(dialog_tokens)
        self.assertEqual(expected, result)

if __name__ == "__main__":
    unittest.main()
