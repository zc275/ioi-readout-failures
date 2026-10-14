"""Shared setup: model loading, paths, seeding."""

import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_DIR, 'ioi_repro', 'outputs')
# Make the authors' modules (causal_interventions, probe, ...) importable.
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

MODEL_ID = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Pin the exact weights so results are reproducible if the hub repo changes.
MODEL_REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer():
  """Loads Llama-3-8B-Instruct in bf16 on the GPU, configured as in the authors' demo."""
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
  # Llama 3 has no pad token. The authors' demo reuses EOS (<|eot_id|>), but
  # their intervention code rebuilds attention masks as `input_ids != pad`, and
  # every IOI prompt contains <|eot_id|> in its chat template, which would then
  # be masked out and change the model's behaviour (see README). So we pad
  # with <|end_of_text|>, which never occurs in the prompts. Padding goes on
  # the left so the last position of every row is the prompt's final token.
  tokenizer.pad_token = '<|end_of_text|>'
  tokenizer.padding_side = 'left'
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
    device_map='cuda', low_cpu_mem_usage=True)
  model.eval()
  for p in model.parameters():
    p.requires_grad_(False)
  return model, tokenizer
