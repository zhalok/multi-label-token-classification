from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from config import id2label
import pandas as pd
import os
from dotenv import load_dotenv
from predict import predict
import evaluate
import json

seqeval = evaluate.load("seqeval")


load_dotenv()

# Load your Hugging Face dataset
print("loading dataset")
test_dataset = pd.read_json(os.path.join("datasets", "test.json"))


predicted_pos_labels = []
predicted_ner_labels = []

true_ner_labels = test_dataset["ner_tags"].tolist()
true_pos_labels = test_dataset["pos_tags"].tolist()


for i in range(len(test_dataset)):
    example = test_dataset.iloc[i]
    text = example["text"]
    predictions = predict(text)
    pos_preds = [prediction["pos_pred"] for prediction in predictions]
    ner_preds = [prediction["ner_pred"] for prediction in predictions]

    predicted_pos_labels.append(pos_preds)
    predicted_ner_labels.append(ner_preds)

results_pos = seqeval.compute(
    predictions=predicted_pos_labels, references=true_pos_labels
)
results_ner = seqeval.compute(
    predictions=predicted_ner_labels, references=true_ner_labels
)

with open("ner_pred_test_results.json", "w") as ner_results_file:
    json.dump(results_ner, ner_results_file)

with open("pos_pred_test_results.json", "w") as pos_results_file:
    json.dump(results_pos, pos_results_file)


# print(all_predicted_ner_labels)
# print(all_predicted_pos_labels)
