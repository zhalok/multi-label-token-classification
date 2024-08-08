from preprocessing.prepare import create_hf_dataset
from preprocessing.tokenize import tokenize_and_align_labels
from trainer import prepare_trainer, train
import pandas as pd
import os


dataset_dir = "datasets"

train_set = pd.read_json(os.path.join(dataset_dir, "train.json"))
val_set = pd.read_json(os.path.join(dataset_dir, "val.json"))

train_set = create_hf_dataset(train_set)
val_set = create_hf_dataset(val_set)

tokenized_train_dataset = train_set.map(tokenize_and_align_labels)
tokenized_validation_dataset = val_set.map(tokenize_and_align_labels)

trainer = prepare_trainer(
    train_set=tokenized_train_dataset, val_set=tokenized_validation_dataset
)
train(trainer=trainer)
