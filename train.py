from preprocessing.preprocess import preprocess_raw_dataset
from preprocessing.prepare import prepare_dataset
from preprocessing.tokenize import tokenize_and_align_labels
from trainer import prepare_trainer, train


preprocess_raw_dataset()
prepared_dataset = prepare_dataset()

train_set = prepared_dataset["train_dataset"]
val_set = prepared_dataset["val_dataset"]
test_set = prepared_dataset["test_dataset"]


tokenized_train_dataset = train_set.map(tokenize_and_align_labels)
tokenized_validation_dataset = val_set.map(tokenize_and_align_labels)

trainer = prepare_trainer(
    train_set=tokenized_train_dataset, val_set=tokenized_validation_dataset
)
train(trainer=trainer)