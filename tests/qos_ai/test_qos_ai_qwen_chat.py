import unittest

import qos_ai as ai
from qos_ai_qwen_chat import CODE_REQUEST, chat_text, loop_start


class ChatFormatTest(unittest.TestCase):
  def test_matches_the_training_format(self):
    self.assertEqual(chat_text('What is 2?'), ai.chat_text('What is 2?'))
    self.assertEqual(chat_text('What is 2?', 'It prints 2.'), ai.chat_text('What is 2?', 'It prints 2.'))


class RetrievalScopeTest(unittest.TestCase):
  def test_program_requests_skip_references(self):
    self.assertTrue(CODE_REQUEST.search('Write a Sol program that prints 3.'))
    self.assertFalse(CODE_REQUEST.search('How do I read a command-line argument in a Sol script?'))


class LoopTest(unittest.TestCase):
  def test_repetition_is_cut_after_one_copy(self):
    row = [1, 2, 3] + [7, 8, 9, 10] * 5
    self.assertEqual(loop_start(row), 7)
    self.assertIsNone(loop_start([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    self.assertIsNone(loop_start([5, 6, 7, 8, 5, 6, 7, 8]))  # two copies are allowed


if __name__ == '__main__':
  unittest.main()
