from preprocessing.preprocess import preprocess_raw_dataset
from preprocessing.prepare import prepare_dataset, resample_dataset, split_dataset
import os


preprocess_raw_dataset()
prepared_dataset = prepare_dataset()

print(type(prepared_dataset))

splited_dataset = split_dataset(prepared_dataset)

train_set = splited_dataset["train"]
test_set = splited_dataset["test"]
val_set = splited_dataset["val"]

train_set = resample_dataset(train_set)
val_set = resample_dataset(val_set)
test_set = resample_dataset(test_set)

train_set.to_json(os.path.join("datasets", "train.json"))
val_set.to_json(os.path.join("datasets", "val.json"))
test_set.to_json(os.path.join("datasets", "test.json"))
