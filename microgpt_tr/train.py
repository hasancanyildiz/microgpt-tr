"""MLP ve Transformer modellerini isim verisiyle eğitir.
Eğitim ayarlarını, ortak tokenizer'ı ve train isimlerini alır; parametreleri
günceller ve her adımın loss değerini runs klasöründeki CSV dosyasına yazar.
"""

import csv
import random
from pathlib import Path

from .mlp import MLP, softmax
from .model import gpt, init_params, new_cache


def train(config, train_names, steps, optimizer, seed):
    """Modeli seçilen optimizer ile eğitir ve parametreleri döndürür."""
    if not train_names:
        raise ValueError("train_names boş olamaz.")
    if steps <= 0:
        raise ValueError("steps pozitif olmalı.")
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer 'sgd' veya 'adam' olmalı.")

    tokenizer = config["tokenizer"]
    model_type = config["model_type"]
    rng = random.Random(seed)

    if model_type == "mlp":
        model = MLP(
            tokenizer.vocab_size,
            n_embd=config["n_embd"],
            seed=seed,
        )
        params = model.state_dict
        parameters = model.params
    elif model_type == "transformer":
        params = init_params(
            config,
            seed=seed,
            vocab_size=tokenizer.vocab_size,
        )
        parameters = [
            parameter
            for matrix in params.values()
            for row in matrix
            for parameter in row
        ]
    else:
        raise ValueError("model_type 'mlp' veya 'transformer' olmalı.")

    learning_rate = config.get("learning_rate", 0.01)
    run_name = config["name"]
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{run_name}.csv"

    beta1 = 0.85
    beta2 = 0.99
    epsilon = 1e-8
    first_moment = [0.0] * len(parameters)
    second_moment = [0.0] * len(parameters)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])

        for step in range(steps):
            name = train_names[step % len(train_names)]
            tokens = tokenizer.encode(name)
            token_count = min(config["block_size"], len(tokens) - 1)
            losses = []

            if model_type == "transformer":
                cache = new_cache(config)

            for position in range(token_count):
                if model_type == "mlp":
                    logits = model(tokens[position])
                else:
                    logits = gpt(
                        tokens[position],
                        position,
                        params,
                        config,
                        cache,
                    )

                probabilities = softmax(logits)
                losses.append(-probabilities[tokens[position + 1]].log())

            loss = sum(losses) / token_count
            loss.backward()

            rate = learning_rate * (1 - step / steps)

            for index, parameter in enumerate(parameters):
                gradient = parameter.grad

                if optimizer == "sgd":
                    parameter.data -= rate * gradient
                else:
                    first_moment[index] = (
                        beta1 * first_moment[index]
                        + (1 - beta1) * gradient
                    )
                    second_moment[index] = (
                        beta2 * second_moment[index]
                        + (1 - beta2) * gradient ** 2
                    )
                    corrected_m = first_moment[index] / (
                        1 - beta1 ** (step + 1)
                    )
                    corrected_v = second_moment[index] / (
                        1 - beta2 ** (step + 1)
                    )
                    parameter.data -= rate * corrected_m / (
                        corrected_v ** 0.5 + epsilon
                    )

                parameter.grad = 0.0

            writer.writerow([step + 1, loss.data])

    return {
    "params": params,
    "model_type": model_type,
    "config": config,
    "tokenizer": tokenizer,
}
