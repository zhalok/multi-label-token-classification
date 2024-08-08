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


def resample_dataset(df):
    dataset = []

    for i in range(len(df)):
        data = df.iloc[i]
        dataset.append({"tokens": data["tokens"], "tags": data["ner_tag_ids"]})
        dataset.append({"tokens": data["tokens"], "tags": data["pos_tag_ids"]})

    resampled_df = pd.DataFrame(dataset)

    return resampled_df


def prepare_dataset():

    current_dir = os.path.dirname(__file__)

    # Construct the path to dataset.json
    dataset_path = os.path.join(current_dir, "..", "datasets", "processed_data.json")
    dataset_dir = os.path.join(current_dir, "..", "datasets")

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
        prepared_df = processed_df.apply(
            lambda x: process_tag_ids(row=x, label2id=label2id), axis=1
        )

        return prepared_df


def create_hf_dataset(df):
    columns = df.columns

    hf_dataset = {"id": [i for i in range(df.shape[0])]}
    for col in columns:
        hf_dataset[col] = df[col].tolist()

    hf_dataset = Dataset.from_dict(hf_dataset)

    return hf_dataset
