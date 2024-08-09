the main strategy that is followed for predicting two type of tags with a single model used here is
the model will be predicting a tensor of size 36 for each token 
first 15 values of the tensor will be predicting the pos and the rest will be predicting ner 

we will split the prediction tensor into two subarrays from 0 to 14 and from 15 to 35 

then we will apply argmax in each of the subarrays to predict both of the tags

the training strategy applied here is to resample each text twice with two set of token tags 

"sample text" : ["pos-tag-1","pos-tag-2"] ["ner-tag-1","ner-tag-2"]

this is how each sample after preprocessing the dataset

after resampling 

"sample text": ["pos-tag-1","pos-tag2"]
"sample text": ["ner-tag-1","ner-tag-2"]

the main intuition behind following this approach is the model should automatically learn to predict the probability of one item in the first 0 to 14 tensor and one item in the second 15 to 35 tensor to be high for each token.

how to run

create a virtual-env
```
python3 -m venv venv
```

activate virtual env 
```
source /venv/bin/activate
```

place the raw tsv data inside datasets/

preprocess the dataset
```
python3 prepare_dataset.py
```

create a .env file
```
tourch .env
```
put environment variable for model name 
```
echo "MODEL_NAME=google-bert/bert-multilingual-cased" > .env
```
train the model
```
python3 train.py
```

put the trained model path into environment variables
```
echo "MODEL_PATH=model_path" >> .env
```

test with testing dataset split

```
python3 test.py
```

predict a single text with inference

```
python3 inference.py
```