"""Shared setup: model loading, paths, seeding."""

import os
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Second-model replication (PLAN_LOCALIZE.md): OOD_MODEL=qwen2.5-7b switches the
# model, and outputs go to ioi_repro/outputs_{tag} so nothing for Llama is touched.
MODEL_TAG = os.environ.get('OOD_MODEL', 'llama3')
OUT_DIR = os.path.join(REPO_DIR, 'ioi_repro', 'outputs' if MODEL_TAG == 'llama3' else f'outputs_{MODEL_TAG}')
# Make the authors' modules (causal_interventions, probe, ...) importable.
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

MODELS = {
  # Pin the exact weights so results are reproducible if the hub repo changes.
  'llama3': ('meta-llama/Meta-Llama-3-8B-Instruct', '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2', '<|end_of_text|>'),
  # Pad with Qwen's <|endoftext|>, which never occurs in the prompts.
  'qwen2.5-7b': ('Qwen/Qwen2.5-7B-Instruct', os.environ.get('QWEN_REVISION', 'main'), '<|endoftext|>'),
}
MODEL_ID, MODEL_REVISION, PAD_TOKEN = MODELS[MODEL_TAG]


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
  tokenizer.pad_token = PAD_TOKEN
  tokenizer.padding_side = 'left'
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
    device_map='cuda', low_cpu_mem_usage=True)
  model.eval()
  register_with_pyvene(model)
  for p in model.parameters():
    p.requires_grad_(False)
  return model, tokenizer


def register_with_pyvene(model):
  """pyvene has no Qwen2 model card. Qwen2's module layout (model.layers[i],
  self_attn, mlp) and config names match Llama's, so reuse Llama's mapping."""
  from pyvene.models import intervenable_modelcard as card
  from transformers.models.llama.modeling_llama import LlamaForCausalLM
  if type(model) not in card.type_to_module_mapping:
    card.type_to_module_mapping[type(model)] = card.type_to_module_mapping[LlamaForCausalLM]
    card.type_to_dimension_mapping[type(model)] = card.type_to_dimension_mapping[LlamaForCausalLM]
