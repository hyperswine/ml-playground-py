import unittest

import qos_ai as ai
from qos_ai_qwen_chat import CODE_REQUEST, chat_text


class ChatFormatTest(unittest.TestCase):
  def test_matches_the_training_format(self):
    self.assertEqual(chat_text('What is 2?'), ai.chat_text('What is 2?'))
    self.assertEqual(chat_text('What is 2?', 'It prints 2.'), ai.chat_text('What is 2?', 'It prints 2.'))


class RetrievalScopeTest(unittest.TestCase):
  def test_program_requests_skip_references(self):
    self.assertTrue(CODE_REQUEST.search('Write a Sol program that prints 3.'))
    self.assertFalse(CODE_REQUEST.search('How do I read a command-line argument in a Sol script?'))


if __name__ == '__main__':
  unittest.main()
