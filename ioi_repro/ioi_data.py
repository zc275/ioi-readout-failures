"""IOI task: data loading and the high-level causal model.

High-level model (Huang et al. 2025, Fig. 1 / Fig. 5):

  name a, name b, name c  ->  position V  ->  output Y

  V = 'first'  if name c == name b   (the non-repeated name is name a)
  V = 'second' if name c == name a   (the non-repeated name is name b)
  Y = name a if V == 'first' else name b

The released data files only contain prompt strings plus a correct/wrong
label (judged against Llama-3-8B-Instruct outputs), so we recover the input
variables by parsing the prompt.
"""

import difflib
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ioi_task')
MODEL_NAME = 'Meta-Llama-3-8B-Instruct'

# Our setting name -> file infix in the released data.
SETTINGS = {
  'id': 'output_wo_quote_in_distribution',
  'ood_chinese': 'ood_chinese_poet_wo_quote_answer',
  'ood_rephrase': 'ood_story_w_prefix_answer',
}
FOLDS = (0, 1, 2)

# Covers all three templates (Table 4 of the paper):
#   ID / Chinese: "...: Then, {a} and {b} were working at the {place}. {c} decided to give a {obj} to"
#   Rephrase:     "Story: {a} and {b} were working at the {place}. {c} decided to give a {obj} to..."
_IOI_PATTERN = re.compile(
  r'(?:Then, |Story: )(?P<a>.+?) and (?P<b>.+?) were working at the (?P<place>.+?)\. '
  r'(?P<c>.+?) decided to give an? (?P<obj>.+?) to')


def load_split(setting, fold, released=False):
  """Returns {'train'|'val'|'test': {'correct': [prompt], 'wrong': [prompt]}}.

  With OOD_MODEL set to another model (common.py), returns that model's own
  splits (15_relabel.py: same train/val/test membership, re-templated prompts,
  correct/wrong from its outputs) unless released=True.
  """
  tag = os.environ.get('OOD_MODEL', 'llama3')
  if tag != 'llama3' and not released:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'outputs_{tag}', 'splits',
                        f'{setting}_fold{fold}.json')
    with open(path) as f:
      return json.load(f)
  path = os.path.join(DATA_DIR, f'ioi_task_{MODEL_NAME}_{SETTINGS[setting]}_split_{fold}.json')
  with open(path) as f:
    return json.load(f)


def parse_ioi_prompt(prompt):
  """Parses a prompt into the high-level model's variables."""
  m = _IOI_PATTERN.search(prompt)
  if m is None:
    raise ValueError(f'Not an IOI prompt: {prompt!r}')
  a, b, c = m.group('a'), m.group('b'), m.group('c')
  if a == b or c not in (a, b):
    raise ValueError(f'Malformed IOI prompt (a={a!r}, b={b!r}, c={c!r})')
  position = 'first' if c == b else 'second'
  return {
    'name_a': a,
    'name_b': b,
    'name_c': c,
    'position': position,
    'answer': a if position == 'first' else b,
    # Character span of the repeated name's second occurrence; the
    # intervention site is the last token of this span (Table 5).
    'name_c_span': m.span('c'),
  }


def intervention_token_index(tokenizer, prompt):
  """Index (in the unpadded token ids, BOS included) of the intervention site:
  the last token of the second occurrence of the repeated name (Table 5)."""
  span_end = parse_ioi_prompt(prompt)['name_c_span'][1]
  enc = tokenizer(prompt, return_offsets_mapping=True)
  # A multi-byte character (e.g. Chinese) can be split over several byte-level
  # tokens that all report the same character span, so take the last match.
  matches = [i for i, (start, end) in enumerate(enc['offset_mapping']) if start < span_end <= end]
  if not matches:
    raise ValueError(f'No token ends the repeated name in {prompt!r}')
  return matches[-1]


def answer_prefix(prompt):
  """Text the model must emit before the name, given how the prompt ends.

  The ID prompt ends with an open quote ('... is "'), so the name follows with
  no space. The OOD prompts end with 'Answer:', so the name follows a space.
  """
  return '' if prompt.endswith('"') else ' '


def output_position(output_text, variables):
  """Which position ('first'/'second') the model's output names.

  Exact (prefix) matches first; otherwise the name the output most resembles.
  The latter covers spelling 'corrections' such as Harrisson -> Harrison.
  Returns (position, exact_match).
  """
  text = output_text.lstrip(' "\'')
  a, b = variables['name_a'], variables['name_b']
  for name in sorted((a, b), key=len, reverse=True):
    if text.startswith(name):
      return ('first' if name == a else 'second'), True
  produced = re.match(r'[^"\'.,\n]*', text).group(0).strip()
  sim = {n: difflib.SequenceMatcher(None, produced, n).ratio() for n in (a, b)}
  return ('first' if sim[a] >= sim[b] else 'second'), False


def counterfactual_label(base_vars, source_vars):
  """y_cf: the base example's name at the source's value of V.

  During DAS training sources are verified (correct) examples, so their
  ground-truth V is also the V their output implies. For counterfactual
  simulation, pass {'position': output_position(...)}: the paper defines
  y_cf from the source's *model output*, y_cf = 1(tau(M(x_src))) (Sec. 4.3),
  and the authors' MMLU demo builds test-time labels from model generations.
  (Our first guess, the ground-truth V, gives AUC ~0.5: see README.)
  """
  if source_vars['position'] == 'first':
    return base_vars['name_a']
  return base_vars['name_b']
