import nltk

import re

from gensim.models import KeyedVectors
from gensim import corpora, models, similarities
nltk.download('stopwords')
from nltk.corpus import stopwords
import jieba



def text_prepare(text):
    """Makes vector model worse"""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()



def load_embeddings():
    word_vectors = KeyedVectors.load("test.word2vec", mmap='r')
    return word_vectors



##get matching availability for individual sentences

def get_most_similar(keyword, texts):
    # normalize instead of split
    texts = [" ".join(text).split() for text in texts.values()]
    dictionary = corpora.Dictionary(texts)
    feature_cnt = len(dictionary.token2id)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    kw_vector = dictionary.doc2bow(jieba.lcut(keyword))
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=feature_cnt)
    sim = index[tfidf[kw_vector]]
    return sim
