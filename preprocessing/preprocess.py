from util import remove_new_line_char
import json


def preprocess_raw_dataset():
    with open("datasets/data.tsv","r") as data_file:
        data = data_file.readlines()
        
        rows = []
        row = []
        for d in data:
            if d != "\n":
                row.append(d)
            else:
                rows.append(row)
                row=[]
        dataset = []

        for row in rows:
            
            sample = {}
            sample["text"] = remove_new_line_char(row[0])
            tokens = []
            ner_tags = []
            pos_tags = []
            for i in range(1,len(row)):
                token, pos_tag, ner_tag = row[i].split("\t")
                token = remove_new_line_char(token)
                pos_tag = remove_new_line_char(pos_tag)
                ner_tag = remove_new_line_char(ner_tag)

                tokens.append(token)

                pos_tags.append(pos_tag)
                ner_tags.append(ner_tag)
            sample["tokens"] = tokens
            sample["ner_tags"] = ner_tags
            sample["pos_tags"] = pos_tags

            dataset.append(sample)
        

        with open("datasets/processed_data.json","w") as json_file:
            json.dump(dataset,json_file)
        
        return dataset
