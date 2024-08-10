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

if you find it difficult to run the code then simply run the code cells in the notebook [Colab_Notebook](https://colab.research.google.com/drive/16zNR_6pbWTdauYF11XBKQ3vKwgKkhS88?usp=sharing)
