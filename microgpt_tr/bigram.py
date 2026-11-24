"""Karakterler arasındaki ikili geçişleri sayar.
Geçiş sayılarını yumuşatılmış sonraki karakter olasılıklarına dönüştürür.
"""


class BigramModel:
    """Tokenizer sözlüğüyle isimlerden bigram modeli öğrenir."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.counts = [
            [0] * self.vocab_size
            for _ in range(self.vocab_size)
        ]

    def fit(self, names):
        self.counts = [
            [0] * self.vocab_size
            for _ in range(self.vocab_size)
        ]

        for name in names:
            tokens = self.tokenizer.encode(name)

            for current, next_token in zip(tokens, tokens[1:]):
                self.counts[current][next_token] += 1

        return self

    def probabilities(self, token_id):
        row = self.counts[token_id]
        total = sum(row) + self.vocab_size

        return [
            (count + 1) / total
            for count in row
        ]
