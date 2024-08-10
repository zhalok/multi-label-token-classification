import evaluate
import torch
import numpy as np
from config import label_list

seqeval = evaluate.load("seqeval")


def remove_new_line_char(string):
    return string.replace("\n", "")


def apply_padding(arr, pad_size):
    padded_arr = arr
    for i in range(pad_size):
        padded_arr.append("PAD")

    return padded_arr


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_predicted_class_label(class_probabilities):
    ...
    pos_class_probs = class_probabilities[:15]
    ner_class_probs = class_probabilities[15:]

    pos_class_pred = torch.argmax(pos_class_probs).item()
    ner_class_pred = torch.argmax(ner_class_probs).item() + 15

    return {"pos_class": pos_class_pred, "ner_class": ner_class_pred}
