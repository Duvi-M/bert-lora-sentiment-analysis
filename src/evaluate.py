import evaluate


def build_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(prediction):
        preds = prediction.predictions.argmax(-1)
        accuracy = accuracy_metric.compute(
            predictions=preds,
            references=prediction.label_ids,
        )["accuracy"]

        return {"accuracy": accuracy}

    return compute_metrics