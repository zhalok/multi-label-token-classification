from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from config import model_path
from util import get_predicted_class_label


def predict(text):

    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    id2label = model.config.id2label

    tokens = text.split(" ")

    inputs = tokenizer(
        tokens, return_tensors="pt", truncation=True, is_split_into_words=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    logit = logits[0]

    word_ids = inputs.word_ids()

    words = []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id != None:
            if word_id != prev_word_id:
                token_pred = get_predicted_class_label(logit[idx])
                pos_pred = token_pred["pos_class"]
                ner_pred = token_pred["ner_class"]
                words.append(
                    {
                        "word": tokens[word_id],
                        "pos_pred": id2label[pos_pred],
                        "ner_pred": id2label[ner_pred],
                    }
                )
            prev_word_id = word_id

    return words
