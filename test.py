from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from config import id2label
import pandas as pd
import os
from dotenv import load_dotenv
from prediction import predict
import evaluate
import json
from util import apply_padding

seqeval = evaluate.load("seqeval")


load_dotenv()

# Load your Hugging Face dataset
print("loading dataset")
test_dataset = pd.read_json(os.path.join("datasets", "test.json"))


predicted_pos_labels = []
predicted_ner_labels = []

true_ner_labels = test_dataset["ner_tags"].tolist()
true_pos_labels = test_dataset["pos_tags"].tolist()

largest_seq_size = max(len(seq) for seq in true_ner_labels)

true_ner_labels = [
    apply_padding(seq, largest_seq_size - len(seq)) for seq in true_ner_labels
]
true_pos_labels = [
    apply_padding(seq, largest_seq_size - len(seq)) for seq in true_pos_labels
]


idx = 0

for i in range(len(test_dataset)):
    example = test_dataset.iloc[i]
    text = example["text"]
    predictions = predict(text)
    pos_preds = [prediction["pos_pred"] for prediction in predictions]
    ner_preds = [prediction["ner_pred"] for prediction in predictions]

    pos_preds = apply_padding(pos_preds, largest_seq_size - len(pos_preds))
    ner_preds = apply_padding(ner_preds, largest_seq_size - len(ner_preds))

    predicted_pos_labels.append(pos_preds)
    predicted_ner_labels.append(ner_preds)


results_pos = seqeval.compute(
    predictions=predicted_pos_labels, references=true_pos_labels
)

results_pos_dict = {
    "precision": results_pos["overall_precision"],
    "recall": results_pos["overall_recall"],
    "f1": results_pos["overall_f1"],
    "accuracy": results_pos["overall_accuracy"],
}


results_ner = seqeval.compute(
    predictions=predicted_ner_labels, references=true_ner_labels
)

results_ner_dict = {
    "precision": results_ner["overall_precision"],
    "recall": results_ner["overall_recall"],
    "f1": results_ner["overall_f1"],
    "accuracy": results_ner["overall_accuracy"],
}


with open("ner_pred_test_results.json", "w") as ner_results_file:
    json.dump(results_ner_dict, ner_results_file)

with open("pos_pred_test_results.json", "w") as pos_results_file:
    json.dump(results_pos_dict, pos_results_file)
