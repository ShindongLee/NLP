# NLP_assignment
Word2vec, FastText, TextClassification - Natural Language Processing (COSE-461 2019-1R)


## How To Use  

### 1. Word2vec
#### Vanilla implementation of Word2vec. It is so slow that you should use 'part' for training data.

python word2vec.py [mode] [training_data_coverage]  
mode: SG/CBOW  
training_data_coverage: part/full  

ex) python word2vec.py SG part  
ex) python word2vec.py CBOW part  

### 2. Word2vec_version2
#### Use Subsampling, Hierachical Softmax, Negative Sampling for reducing training time.  

python word2vec.py [mode] [number_of_negative_sampling] [training_data_coverage]  
mode: SG/CBOW  
number_of_negative_sampling: 0 for Hierachical Softmax / Positive integer for negative sampling  
training_data_coverage: part/full  

ex) python word2vec.py SG 0 part  
ex) python word2vec.py CBOW 20 full  

### 3. FastText
#### Implementation of FastText

python fasttext.py [number_of_negative_sampling] [training_data_coverage]  
number_of_negative_sampling: Positive integer for negative sampling  
training_data_coverage: part/full  

ex) python fasttext.py SG 0 part
ex) python fasttext.py CBOW 20 full

### 4. Text Classification  
#### Implementation of FastText Text Classification  

python classify.py [use_bigram] [dimension] [epoch] [learning_rate]  
use_bigram: True/False  
dimension: dimension of embedding vector  
epoch: total epoch for training  
learning_rate: eta  

ex) python classify.py True 7 0.01 
