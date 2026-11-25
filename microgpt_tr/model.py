"""Autograd ile karakter düzeyinde Transformer hesaplarını içerir.
Parametre yapılandırması ve KV cache alıp bir sonraki token için logit üretir.
"""

import random

from .mlp import linear, softmax
from .value import Value


def rmsnorm(values):
    mean_square = sum(value * value for value in values) / len(values)
    scale = (mean_square + 1e-5) ** -0.5
    return [value * scale for value in values]


def init_params(config, seed=42, vocab_size=30):
    """Yapılandırmaya göre deterministik model parametreleri üretir."""
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    block_size = config["block_size"]

    if n_embd % n_head != 0:
        raise ValueError("n_embd, n_head sayısına tam bölünmeli.")

    rng = random.Random(seed)

    def matrix(rows, cols):
        return [
            [Value(rng.gauss(0, 0.08)) for _ in range(cols)]
            for _ in range(rows)
        ]

    params = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }

    for layer in range(n_layer):
        prefix = f"layer{layer}"
        params[f"{prefix}.attn_wq"] = matrix(n_embd, n_embd)
        params[f"{prefix}.attn_wk"] = matrix(n_embd, n_embd)
        params[f"{prefix}.attn_wv"] = matrix(n_embd, n_embd)
        params[f"{prefix}.attn_wo"] = matrix(n_embd, n_embd)
        params[f"{prefix}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        params[f"{prefix}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

    return params


def new_cache(config):
    """Her Transformer katmanı için boş key/value cache oluşturur."""
    n_layer = config["n_layer"]
    return {
        "keys": [[] for _ in range(n_layer)],
        "values": [[] for _ in range(n_layer)],
    }


def gpt(token_id, pos_id, params, config, cache):
    """Bir token ve konum için Transformer logitlerini hesaplar."""
    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    block_size = config["block_size"]

    if pos_id >= block_size:
        raise ValueError("Konum block_size sınırını aşamaz.")

    head_dim = n_embd // n_head
    x = [
        token + position
        for token, position in zip(
            params["wte"][token_id],
            params["wpe"][pos_id],
        )
    ]

    for layer in range(n_layer):
        prefix = f"layer{layer}"
        residual = x
        x = rmsnorm(x)

        query = linear(x, params[f"{prefix}.attn_wq"])
        key = linear(x, params[f"{prefix}.attn_wk"])
        value = linear(x, params[f"{prefix}.attn_wv"])

        cache["keys"][layer].append(key)
        cache["values"][layer].append(value)

        attended = []
        for head in range(n_head):
            start = head * head_dim
            end = start + head_dim
            query_head = query[start:end]
            keys_head = [item[start:end] for item in cache["keys"][layer]]
            values_head = [
                item[start:end] for item in cache["values"][layer]
            ]

            attention_logits = [
                sum(query_head[i] * keys_head[t][i] for i in range(head_dim))
                / (head_dim ** 0.5)
                for t in range(len(keys_head))
            ]
            weights = softmax(attention_logits)

            head_output = [
                sum(
                    weights[t] * values_head[t][i]
                    for t in range(len(values_head))
                )
                for i in range(head_dim)
            ]
            attended.extend(head_output)

        x = linear(attended, params[f"{prefix}.attn_wo"])
        x = [left + right for left, right in zip(x, residual)]

        residual = x
        x = rmsnorm(x)
        x = linear(x, params[f"{prefix}.mlp_fc1"])
        x = [value.relu() for value in x]
        x = linear(x, params[f"{prefix}.mlp_fc2"])
        x = [left + right for left, right in zip(x, residual)]

    return linear(x, params["lm_head"])
