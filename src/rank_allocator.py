from collections import defaultdict


def extract_layer_key(param_name: str) -> str:
    name = param_name
    name = name.replace("base_model.model.", "")
    name = name.replace(".lora_A.default.weight", "")
    name = name.replace(".lora_B.default.weight", "")
    return name


def collect_gradient_layer_scores(model, dataloader, max_steps: int, device: str):
    model.train()
    model.to(device)

    layer_scores = defaultdict(float)
    steps = 0

    for batch in dataloader:
        if steps >= max_steps:
            break

        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

        # Hugging Face espera "labels", pero el dataset IMDb trae "label"
        if "label" in batch and "labels" not in batch:
            batch["labels"] = batch.pop("label")

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if loss is None:
            raise ValueError(
                "Loss is None. Revisa que el batch tenga la clave 'labels'."
            )

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            if "lora_A" in name or "lora_B" in name:
                layer_key = extract_layer_key(name)
                layer_scores[layer_key] += param.grad.detach().norm().item()

        steps += 1

    if steps > 0:
        for layer in layer_scores:
            layer_scores[layer] /= steps

    return dict(layer_scores)


def normalize_scores(layer_scores: dict):
    if not layer_scores:
        return layer_scores

    max_score = max(layer_scores.values())

    if max_score == 0:
        return {layer: 1.0 for layer in layer_scores}

    return {
        layer: score / max_score
        for layer, score in layer_scores.items()
    }


def allocate_ranks(layer_scores, total_budget, min_rank=2, max_rank=8, step=2):
    rank_pattern = {
        layer_name: min_rank
        for layer_name in layer_scores
    }

    used_budget = len(rank_pattern) * min_rank
    history = []

    while used_budget + step <= total_budget:
        best_layer = None
        best_score = -1

        for layer_name, score in layer_scores.items():
            current_rank = rank_pattern[layer_name]

            if current_rank + step <= max_rank and score > best_score:
                best_score = score
                best_layer = layer_name

        if best_layer is None:
            break

        rank_pattern[best_layer] += step
        used_budget += step

        history.append({
            "iteration": len(history) + 1,
            "selected_layer": best_layer,
            "score": best_score,
            "new_rank": rank_pattern[best_layer],
            "used_budget": used_budget,
        })

    return rank_pattern, history
