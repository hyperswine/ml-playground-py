import unittest

from qos_ai_qwen_sft import micro_batches, steps_first


def example(length, targets):
  return ([1] * length, [False] * (length - targets) + [True] * targets)


class MicroBatchTest(unittest.TestCase):
  def test_every_example_once_within_budgets(self):
    examples = [example(n, t) for n, t in [(120, 40), (1833, 300), (300, 90), (90, 30), (1500, 200), (512, 400),
                                            (200, 60), (60, 20)] * 4]
    members = list(range(len(examples)))
    batches = micro_batches(members, examples, token_budget=2048, target_budget=1024)
    self.assertEqual(sorted(i for b in batches for i in b), members)
    for b in batches:
      width = max(len(examples[i][0]) for i in b)
      if len(b) > 1:
        self.assertLessEqual(len(b) * width, 2048)
        self.assertLessEqual(sum(sum(examples[i][1][1:]) for i in b), 1024)

  def test_long_examples_train_alone(self):
    examples = [example(1833, 300)] * 4
    self.assertEqual([len(b) for b in micro_batches([0, 1, 2, 3], examples)], [1, 1, 1, 1])


class StepsFirstTest(unittest.TestCase):
  def test_steps_move_before_the_claim(self):
    self.assertEqual(steps_first('It prints 75. Steps: filter=[8, 12, 5]; sum=75.'),
                     'Steps: filter=[8, 12, 5]; sum=75.\nIt prints 75.')
    code = '```sol\n> print (str (1 + 2)).\n```\nIt prints 3. Steps: add=3.'
    self.assertEqual(steps_first(code), '```sol\n> print (str (1 + 2)).\n```\nSteps: add=3.\nIt prints 3.')
    self.assertEqual(steps_first('No change needed. It prints [a. b]. Steps: x=1.'),
                     'No change needed. Steps: x=1.\nIt prints [a. b].')

  def test_other_answers_are_left_alone(self):
    self.assertIsNone(steps_first('It prints 3.'))
    self.assertIsNone(steps_first('Option is for values that may be missing.'))


if __name__ == '__main__':
  unittest.main()
