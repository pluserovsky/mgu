from collections import Counter
import numpy as np
import nltk
import re
import sklearn.manifold
import multiprocessing
import pandas as pd
import gensim.models.word2vec as w2v


data = pd.read_csv("../data/wine-reviews/winemag-data_first150k.csv", index_col=False)

labels = data['variety']
descriptions = data['description']

#print('{}   :   {}'.format(labels.tolist()[0], descriptions.tolist()[0]))

varietal_counts = labels.value_counts()
print(varietal_counts[:50])

corpus_raw = ""
for description in descriptions:
    corpus_raw += description

#nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[234])
print(sentence_to_wordlist(raw_sentences[234]))

token_count = sum([len(sentence) for sentence in sentences])
print('The wine corpus contains {0:,} tokens'.format(token_count))

num_features = 300
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993

wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

wine2vec.build_vocab(sentences)

print('Word2Vec vocabulary length:', len(wine2vec.wv.vocab))
print(wine2vec.corpus_count)
wine2vec.train(sentences, total_examples=wine2vec.corpus_count, epochs=wine2vec.iter)

def points_to_ranking(points):
    if points in range(80,83):
        return 0
    elif points in range(83,87):
        return 1
    elif points in range(87,90):
        return 2
    elif points in range(90,94):
        return 3
    elif points in range(94-98):
        return 4
    else:
        return 5
    
#data["rating"] = data["points"].apply(points_to_ranking)
#print data.rating.value_counts()