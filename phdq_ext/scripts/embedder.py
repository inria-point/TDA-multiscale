"""Per-token embeddings from ModernBERT for qPHD point clouds."""
import hashlib
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "answerdotai/ModernBERT-base"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "embeds")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name=MODEL_NAME, device=None, max_length=8192):
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model_name = model_name
        self.max_length = max_length

    @torch.no_grad()
    def embed(self, text, return_tokens=False):
        """Return (n_tokens, hidden_dim) array of last-layer token embeddings.

        Special tokens ([CLS]/[SEP]) are excluded so the point cloud
        contains only real text tokens.
        """
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state[0]  # (seq, dim)
        ids = enc["input_ids"][0]
        special = torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                ids.tolist(), already_has_special_tokens=True
            ),
            dtype=torch.bool,
            device=self.device,
        )
        embeds = out[~special].float().cpu().numpy()
        if return_tokens:
            tokens = self.tokenizer.convert_ids_to_tokens(ids[~special].tolist())
            return embeds, tokens
        return embeds

    def embed_cached(self, text, cache_key=None):
        """Embed with on-disk cache (npz keyed by md5 of model+text)."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        key = cache_key or hashlib.md5(
            (self.model_name + "\x00" + text).encode()
        ).hexdigest()
        path = os.path.join(CACHE_DIR, key + ".npz")
        if os.path.exists(path):
            return np.load(path)["embeds"]
        embeds = self.embed(text)
        np.savez_compressed(path, embeds=embeds)
        return embeds
