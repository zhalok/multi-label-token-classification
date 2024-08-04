from transformers import AutoTokenizer
from config import model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(example):
    # Tokenize the input
    tokenized_input = tokenizer(
        example["tokens"], truncation=True, is_split_into_words=True
    )

    # Initialize list to store labels
    label_ids = []

    # Get the word ids for each token
    word_ids = tokenized_input.word_ids()  # Map tokens to their respective word

    # Initialize previous word index to check token alignment
    previous_word_idx = None

    # Process each token's word index
    for word_idx in word_ids:
        if word_idx is None:
            # If the word index is None (special tokens like [CLS], [SEP], etc.), assign -100
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # Label the first token of a given word
            label_ids.append(example["tags"][word_idx])
        else:
            # Assign -100 to tokens that are not the first token of a word
            label_ids.append(-100)
        # Update the previous word index
        previous_word_idx = word_idx

    # Add labels to the tokenized inputs
    tokenized_input["labels"] = label_ids

    return tokenized_input
