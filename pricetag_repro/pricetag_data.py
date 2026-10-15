"""PriceTag task: data loading, prompt parsing and output judging.

High-level model (Huang et al. 2025, Fig. 2; Wu et al. 2023):

  lower bound lb, upper bound ub, query q
    ->  P = [lb < q]  (left-boundary check)
    ->  Q = [q < ub]  (right-boundary check)
    ->  output Y = P and Q   ("Yes" / "No")

Appendix Fig. 6 notes a shortcut: if P is False the model answers No
without checking Q.

The released files contain prompts plus a correct/wrong label judged against
Llama-3-8B-Instruct's free-form answers, so we recover lb, ub, q by parsing.

Settings (Table 3 of the paper):
  id            "Does the following item cost between $9.32 and $12.1? Item: $7.59."
  ood_currency  same template, Turkish lira with a decimal comma: "₺5,44"
                (file infix "ood_arabic"; the paper's "Change Currency Format")
  ood_typo      typos in the *template* only, numbers unchanged:
                "Does the item cost b/t $$1.98 && $18.4?!! Item: $15.3." with
                an "Answer:" prefix after the chat template ("Add Typos")
"""

import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pricetag_task')
MODEL_NAME = 'Meta-Llama-3-8B-Instruct'

SETTINGS = {
  'id': 'in_distribution',
  'ood_currency': 'ood_arabic',
  'ood_typo': 'ood_typo',
}
FOLDS = (0, 1, 2)

_USER_TEXT = re.compile(r'user<\|end_header_id\|>\n\n(?P<text>.*?)<\|eot_id\|>', re.S)
_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
_VERDICT = re.compile(r'\b(yes|no)\b', re.I)


def load_split(setting, fold):
  """Returns {'train'|'val'|'test': {'correct': [prompt], 'wrong': [prompt]}}."""
  path = os.path.join(DATA_DIR, f'pricetag_task_{MODEL_NAME}_{SETTINGS[setting]}_split_{fold}.json')
  with open(path) as f:
    return json.load(f)


def parse_prompt(prompt):
  """The high-level model's input and intermediate variables for one prompt."""
  text = _USER_TEXT.search(prompt).group('text')
  numbers = _NUMBER.findall(text)
  if len(numbers) != 3:
    raise ValueError(f'Expected 3 numbers in {text!r}, found {numbers}')
  lb, ub, q = (float(n.replace(',', '.')) for n in numbers)
  P, Q = lb < q, q < ub
  return {'lb': lb, 'ub': ub, 'q': q, 'P': P, 'Q': Q, 'answer': P and Q,
          'q_text': numbers[2]}


def judge(output_text, variables):
  """('correct' | 'wrong' | 'undecided', said) for a decoded answer.

  The verdict is the first standalone "yes" or "no" (any case), following the
  paper's answer tokens ("all case variations of yes and no"). No such word
  means the answer is undecided, which the paper excludes as invalid.
  """
  m = _VERDICT.search(output_text)
  if m is None:
    return 'undecided', None
  said = m.group(1).lower() == 'yes'
  return ('correct' if said == variables['answer'] else 'wrong'), said
