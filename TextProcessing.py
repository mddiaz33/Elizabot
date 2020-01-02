import nltk
from nltk.stem import WordNetLemmatizer

# extra things I made while confused

class TextPreProcessor:
    def __init__(self):
        self.tokenizer = nltk.TreebankWordTokenizer()
        self.stemmer = nltk.stem.SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def process_text(self, text):
        t = self.tokenize_text(text.lower())
        n = self.stem_tokens(t)
        t = self.tokenize_text(n)
        return t

    def tokenize_text(self, text):
        text_tokenized_by_language_rules = self.tokenizer.tokenize(text)
        return text_tokenized_by_language_rules

    # try both stemmer and porter and choose one with best results
    def test_normalizers(self, text):
        t = self.tokenize_text(text)
        return self.stem_tokens(t), self.lem_tokens(t)  # add additional text normalization methods as needed

    # token stemmer removes and replaces suffixes to return root of word
    def stem_tokens(self, tokens):
        stemmed_tokens = " ".join(self.stemmer.stem(token) for token in tokens)
        return stemmed_tokens

    # token Lemmatizer uses WordNet to find word lemmas
    def lem_tokens(self, tokens):
        lem_tokens = " ".join(self.lemmatizer.lemmatize(token) for token in tokens)
        return lem_tokens


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd



class FeatureExtractor:
    def __init__(self):
        # a high score indicates high documents frequency and low document co-occurence
        self.tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))  # using unigrams and bigreams

    def get_bag_of_words(self, list_of_texts):
        # get word tf-idf vectors
        features = self.tfidf.fit_transform(list_of_texts)
        # create TF-IDF BOW, normalized to 1 and low frequency bigrams and unigrams omitted
        tfidf_bow = pd.DataFrame(
            features.todense(),
            columns=self.tfidf.get_feature_names()
        )
        return tfidf_bow



if __name__ == '__main__':
    tp = TextPreProcessor()
    print(tp.process_text("Feet cat wolves talked?"))
    texts = ["good movie", "not a good movie", "did not like", "i like it", "good one"]
    fe = FeatureExtractor()
    print( fe.get_bag_of_words(texts))