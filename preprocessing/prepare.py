import json
import pandas as pd
from datasets import Dataset
from preprocessing.splitting import split_dataset
import os


def process_tag_ids(row, label2id):
    ner_tags = row["ner_tags"]
    pos_tags = row["pos_tags"]
    ner_ids = []
    pos_ids = []
    for tag in ner_tags:
        ner_ids.append(label2id[tag])
    for tag in pos_tags:
        pos_ids.append(label2id[tag])

    row["ner_tag_ids"] = ner_ids
    row["pos_tag_ids"] = pos_ids
    return row


def prepare_hf_dataset(df):
    dataset = []

    for i in range(len(df)):
        data = df.iloc[i]
        dataset.append({"tokens": data["tokens"], "tags": data["ner_tag_ids"]})
        dataset.append({"tokens": data["tokens"], "tags": data["pos_tag_ids"]})

    dataset = Dataset.from_dict(
        {
            "id": [i for i in range(len(dataset))],
            "tokens": [d["tokens"] for d in dataset],
            "tags": [d["tags"] for d in dataset],
        }
    )

    return dataset


def prepare_dataset():

    current_dir = os.path.dirname(__file__)

    # Construct the path to dataset.json
    dataset_path = os.path.join(current_dir, "..", "datasets", "preprocessed_data.json")

    with open(dataset_path, "r") as json_file:
        dataset = json.load(json_file)
        processed_df = pd.DataFrame(dataset)
        ner_tags_list = processed_df["ner_tags"].tolist()
        pos_tag_list = processed_df["pos_tags"].tolist()

        unique_ner_tags = set([tag for tags in ner_tags_list for tag in tags])
        unique_pos_tags = set([tag for tags in pos_tag_list for tag in tags])

        unique_tags = list(unique_pos_tags) + list(unique_ner_tags)

        id2label = {i: label for i, label in enumerate(unique_tags)}
        label2id = {label: i for i, label in enumerate(unique_tags)}
        processed_df = processed_df.apply(
            lambda x: process_tag_ids(row=x, label2id=label2id), axis=1
        )
        splitting = split_dataset(processed_df)
        train_df = splitting["train"]
        test_df = splitting["test"]
        val_df = splitting["val"]

        train_dataset = prepare_hf_dataset(train_df)
        val_dataset = prepare_hf_dataset(val_df)
        test_dataset = prepare_hf_dataset(test_df)

        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "val_dataset": val_dataset,
            "label2id": label2id,
            "id2label": id2label,
        }
