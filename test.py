from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from config import id2label
import pandas as pd
import os
from dotenv import load_dotenv
from predict import predict

load_dotenv()

# Load your Hugging Face dataset
test_dataset = pd.read_csv(os.path.join("datasets", "test.json"))

model_path = os.getenv("MODEL_PATH")
# Load your pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)


for example in test_dataset:
    tokens = example["tokens"]
    predictions = predict(tokens)

    # Print or store the results as needed
    print(predictions)
