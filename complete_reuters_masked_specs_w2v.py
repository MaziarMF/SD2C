import tensorflow as tf
import numpy as np
import math
from os.path import isfile
from utils import read_list
from nltk.corpus import reuters
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import PorterStemmer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import itertools


# Fetch the dataset
category_dict = {'acq':0, 'coffee':1, 'crude':2, 'earn':3, 'gold':4, 'interest':5, 'money-fx':6, 'ship':7, 'sugar':8,
                 'trade':9}
data = []
target = []
docs = reuters.fileids()
for doc in docs:
    # Check if the document is only related to 1 class and that class is in category_dict
    if len(reuters.categories(doc)) == 1 and reuters.categories(doc)[0] in category_dict:
        data.append(" ".join(reuters.words(doc))) # Text of the document
        target.append(category_dict[reuters.categories(doc)[0]]) # Index for the class
print("Dataset REUTERS loaded...")

# Pre-process the dataset
print("Pre-processing the dataset...")
stemmer = PorterStemmer() # Define the type of stemmer to use
additional_stop_words = []
stop_words = ENGLISH_STOP_WORDS.union(additional_stop_words)
stop_words = set([stemmer.stem(word) for word in stop_words]) # Stem the stop words for larger detection
processed_data = []
id_to_delete = []
for i, doc in enumerate(data):
    tokenized_doc = list(simple_preprocess(doc, deacc=True, min_len=2))
    stemmed_doc = []
    for word in tokenized_doc:
        stemmed_word = stemmer.stem(word)
        if stemmed_word not in stop_words:
            stemmed_doc.append(stemmed_word)
    #[stemmer.stem(word) for word in tokenized_doc if word not in stop_words]
    if stemmed_doc == []: # Empty document after pre-processing: to be removed
        id_to_delete.append(i)
    else:
        processed_data.append(stemmed_doc)
data = processed_data
target = np.delete(target, id_to_delete, axis=0)
#####
keywords = read_list("constraints/reuters/keywords_freq_auto_5", "str")
keywords = [keywords[i].split(" ") for i in range(len(keywords))] # Otherwise don't stem
nr_kw_perclass = 3
kw= np.array(list(itertools.chain(*keywords)))
counter = np.zeros((len(data), len(kw)))
for i in range(len(data)):
  for k in range(len(data[i])):
    for j in range(len(kw)):
            if kw[j]==data[i][k]:
                counter[i][j] = counter[i][j]+1
print(len(data))
window = 50
model_path = "models/reuters_w2v_window" + str(window) + ".model"
if isfile(model_path): # Load if the word2vec model exists
    print("Loading an existing word2vec model trained on the dataset...")
    w2v = Word2Vec.load(model_path)
else: # Otherwise train the word2vec model and save it
    print("Training a word2vec model on the dataset...")
    w2v = Word2Vec(sentences=data, min_count=1, workers=4, sg=1, window=window) # Train a word2vec model on the data
    w2v.save(model_path)
print("Building word2vec-based representations of the documents...")
data_temp = [np.mean([w2v[word] for word in doc if word in w2v], axis=0) for doc in data] # Average word embeddings in each document
data = np.asarray(data_temp)

# Get the split between training/test set and validation set
test_indices = read_list("split/reuters/test.txt")
validation_indices = read_list("split/reuters/validation.txt")

# Fetch keywords for each class
keywords_tfidf = [[w2v[keyword] for keyword in class_ if keyword in w2v] for class_ in keywords]
keywords_tfidf= np.array(list(itertools.chain(*keywords_tfidf)))
keywords_w2v = [np.mean([w2v[keyword] for keyword in class_ if keyword in w2v], axis=0) for class_ in keywords]
keywords = np.array(keywords_w2v)
###masking operation    
data_sec = []
data_masked = []
#delete_axis = 20
#counter = np.delete(counter, delete_axis, 1)
for i in range(len(data_temp)):
    if(np.count_nonzero(counter[i])!=0):
        data_sec.append(data_temp[i])
        temp = np.dot(np.reshape(counter[i], (1, counter.shape[1])), keywords_tfidf)/np.sum(counter[i])
        data_masked.append(temp)
n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = len(set(target)) # Number of clusters to obtain

########
import pickle
with open("data/reuters/target_reuters.txt", "wb") as fp:
    pickle.dump(target, fp)
with open("data/reuters/data_reuters.txt", "wb") as fp:
    pickle.dump(data, fp)
with open("data/reuters/data_reuters_masked.txt", "wb") as fp:
    pickle.dump(data_masked, fp)
with open("data/reuters/data_reuters_sec.txt", "wb") as fp:
    pickle.dump(data_sec, fp)
#data_second = np.reshape(data_second, (data_second.shape[0], data_second.shape[2]))
with open("data/reuters/keywords_reuters.txt", "wb") as fp:
    pickle.dump(keywords, fp)



########

# Layer sizes
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = 10

dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names