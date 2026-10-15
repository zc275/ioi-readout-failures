"""Amplify a DAS variable at its site with forward hooks (PLAN_AMPLIFY.md, lead 1).

At each hooked layer, for each row, the 1-d DAS coordinate c = h . w at the
row's site token is replaced by m + alpha (c - m), where m is the midpoint of
the two class means of c on ID training prompts. No label of the test prompt
is used.

Random control: the same per-row displacement norm |(alpha - 1)(c - m)|, along a
fixed random unit direction r orthogonal to w, signed by the row's own
coordinate on r (sign(h . r - m_r)).

Hooks act on the decoder layer's output, which is pyvene's 'block_output'.
"""

import numpy as np
import torch


def load_directions(path):
  """{layer: unit vector (d,)} from a saved dim-1 DAS run."""
  weights = torch.load(path, map_location='cuda')
  out = {}
  for k, w in weights.items():
    assert w.shape[0] == 1, f'{k}: amplification needs a dim-1 subspace, got {tuple(w.shape)}'
    out[int(k.split('.')[1])] = (w[0] / w[0].norm()).float().cuda()
  return out


def random_directions(directions, seed):
  """{layer: unit vector orthogonal to the DAS direction}, from a fixed seed."""
  g = torch.Generator(device='cpu').manual_seed(seed)
  out = {}
  for layer, w in directions.items():
    r = torch.randn(w.shape[0], generator=g).cuda()
    r = r - (r @ w) * w
    out[layer] = r / r.norm()
  return out


class Amplifier:
  def __init__(self, model, directions):
    """directions: {layer: DAS unit vector}. Centers start at 0; set with fit_centers."""
    self.w = directions
    self.m = {layer: 0.0 for layer in directions}
    self.r, self.m_r = None, None
    self.alpha, self.positions, self.record = 1.0, None, None
    # With gate on, amplify only rows whose coordinate lies on the gate_sign[layer]
    # (+1 / -1) side of m, i.e. the side the False class sits on.
    self.gate, self.gate_sign = False, {}
    self.handles = [model.model.layers[layer].register_forward_hook(self._hook(layer))
                    for layer in directions]

  def remove(self):
    for h in self.handles:
      h.remove()

  def set_random(self, r, m_r):
    """Switch to the random-direction control (r=None switches back to DAS)."""
    self.r, self.m_r = r, m_r

  def _hook(self, layer):
    def hook(module, inputs, output):
      if self.positions is None:
        return output
      h = output[0]
      rows = torch.arange(h.shape[0], device=h.device)
      x = h[rows, self.positions].float()
      w = self.w[layer]
      c = x @ w
      if self.record is not None:
        coords = [c] + ([x @ self.r[layer]] if self.r is not None else [])
        self.record[layer].append(torch.stack(coords, -1).cpu())
      if self.alpha == 1.0:
        return output
      delta = (self.alpha - 1.0) * (c - self.m[layer])
      if self.gate:
        delta = delta * ((c - self.m[layer]) * self.gate_sign[layer] > 0)
      if self.r is None:
        x = x + delta[:, None] * w
      else:
        r = self.r[layer]
        sign = torch.sign(x @ r - self.m_r[layer])
        x = x + (sign * delta.abs())[:, None] * r
      h[rows, self.positions] = x.to(h.dtype)
      return output
    return hook

  def start_recording(self):
    """Record coordinates on the next forward passes (use with alpha = 1)."""
    self.record = {layer: [] for layer in self.w}

  def stop_recording(self):
    """{layer: (n, 1 or 2)} coordinates on (DAS, [random]) since start_recording."""
    rec = {layer: torch.cat(v).numpy() for layer, v in self.record.items()}
    self.record = None
    return rec


def midpoint(coords, classes):
  """Midpoint of the two class means of a 1-d coordinate."""
  classes = np.asarray(classes, dtype=bool)
  return float((coords[classes].mean() + coords[~classes].mean()) / 2)
