import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import qos_ai as ai
from qos_ai_qwen_chat import CODE_REQUEST, chat_text, loop_start, looping, load, with_references


class ChatFormatTest(unittest.TestCase):
  def test_matches_the_training_format(self):
    self.assertEqual(chat_text('What is 2?'), ai.chat_text('What is 2?'))
    self.assertEqual(chat_text('What is 2?', 'It prints 2.'), ai.chat_text('What is 2?', 'It prints 2.'))


class RetrievalScopeTest(unittest.TestCase):
  def test_program_requests_skip_references(self):
    self.assertTrue(CODE_REQUEST.search('Write a Sol program that prints 3.'))
    self.assertFalse(CODE_REQUEST.search('How do I read a command-line argument in a Sol script?'))


class NewInterfaceTest(unittest.TestCase):
  def test_mlx_adapter_is_forwarded(self):
    backend=Mock(return_value=('model','tokenizer'))
    with patch.dict('sys.modules',{'mlx_lm':SimpleNamespace(load=backend)}):
      self.assertEqual(load('parent','auto',engine='mlx',adapter='candidate'),('model','tokenizer','mlx'))
    backend.assert_called_once_with('parent',adapter_path='candidate')
    with self.assertRaisesRegex(ValueError,'only with --engine mlx'):
      load('parent','cpu',engine='torch',adapter='candidate')

  def test_source_profile_routes_module_requests_and_preserves_plain_code(self):
    class Tok:
      def encode(self,text,**kwargs):return list(map(ord,text))
      def decode(self,ids):return ''.join(map(chr,ids))
    with tempfile.TemporaryDirectory() as directory:
      corpus=Path(directory)/'corpus.jsonl'
      corpus.write_text(json.dumps({'path':'fprisc/sol/lib/json.sol','authority':'implementation',
        'text':'# File: fprisc/sol/lib/json.sol\nparse returns Result Json.'})+'\n')
      args=SimpleNamespace(retrieve=1,corpus=corpus,embeddings=None,reference_tokens=2000,retrieval_profile='source')
      qs=['Write an explanation of the sol/lib/json.sol API.','Write a Sol program that prints 3.']
      prompts,sources=with_references(qs,Tok(),args)
      self.assertIn('Source: <file path>',prompts[0])
      self.assertEqual(prompts[0].count('# File:'),1)
      self.assertEqual(sources[0],['fprisc/sol/lib/json.sol'])
      self.assertEqual(prompts[1],qs[1])
      args.retrieval_profile='legacy'
      self.assertEqual(with_references(qs,Tok(),args)[0],qs)


class LoopTest(unittest.TestCase):
  def test_repetition_is_cut_after_one_copy(self):
    row = [1, 2, 3] + [7, 8, 9, 10] * 5
    self.assertEqual(loop_start(row), 7)
    self.assertIsNone(loop_start([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    self.assertIsNone(loop_start([5, 6, 7, 8, 5, 6, 7, 8]))  # two copies are allowed

  def test_early_stop_rule_needs_four_copies(self):
    self.assertTrue(looping([9, 8, 7, 6, 5, 4, 3, 2] + [1, 2, 3, 4] * 4))  # checked every 8 tokens
    self.assertFalse(looping([9, 8, 7, 6, 5, 4, 3, 2] + [1, 2, 3, 4] * 4 + [5]))
    self.assertFalse(looping(list(range(24))))


if __name__ == '__main__':
  unittest.main()
