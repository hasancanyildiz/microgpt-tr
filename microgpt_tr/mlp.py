"""Autograd tabanlı çok katmanlı algılayıcı modelini içerir.
Token kimliği ve parametre tablolarını alıp sözlük boyutunda logit listesi döndürür.
"""

from .value import Value


def linear(x, weights):
    """Ağırlık matrisini vektöre uygular."""
    return [
        sum(weight * value for weight, value in zip(row, x))
        for row in weights
    ]


def softmax(logits):
    """Logitleri sayısal olarak kararlı olasılıklara dönüştürür."""
    max_value = max(value.data for value in logits)
    exponentials = [(value - max_value).exp() for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class MLP:
    """Tek token için karakter tahmin logitleri üreten MLP."""

    def __init__(self, vocab_size, n_embd=16, seed=42):
        import random

        rng = random.Random(seed)
        matrix = lambda rows, cols: [
            [Value(rng.gauss(0, 0.08)) for _ in range(cols)]
            for _ in range(rows)
        ]

        self.state_dict = {
            "wte": matrix(vocab_size, n_embd),
            "mlp_fc1": matrix(4 * n_embd, n_embd),
            "mlp_fc2": matrix(vocab_size, 4 * n_embd),
        }
        self.params = [
            parameter
            for weights in self.state_dict.values()
            for row in weights
            for parameter in row
        ]

    def __call__(self, token_id):
        x = self.state_dict["wte"][token_id]
        x = linear(x, self.state_dict["mlp_fc1"])
        x = [value.relu() for value in x]
        return linear(x, self.state_dict["mlp_fc2"])
