from csv import DictReader, DictWriter

import numpy as np
from numpy import array
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, hstack
kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1,3), max_df=1.0, min_df=1, vocabulary=None)        
        self.vectorizer2 = CountVectorizer()
    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def train_feature2(self, examples):
        return self.vectorizer2.fit_transform(examples)

    def test_feature2(self, examples):
        return self.vectorizer2.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    extra_info = list(DictReader(open("../data/spoilers/meta_info.csv", 'r')))

    feat = Featurizer()
    # generate a mapping from page to genre
    genre_dic = defaultdict(str)
    for line in extra_info:
        genre_dic[line['page']]=line['Genre']
    # turn the page feature to genre feature
    for line in train:
        if line['page'] in genre_dic:
            line['page'] = genre_dic[line['page']]
        else:
            line['page'] = 'UNK'
        
    for line in test:
        if line['page'] in genre_dic:
            line['page'] = genre_dic[line['page']]
        else:
            line['page'] = 'UNK'

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    # use two count vectorizer to generate feature vector and use hastack to
    # concatenate it
    x_train_sentence = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_train_genre = feat.train_feature2(x['page'] for x in train)
    x_train = hstack((x_train_sentence,x_train_genre))

    x_test_sentence = feat.test_feature(x[kTEXT_FIELD] for x in test)
    x_test_genre = feat.test_feature2(x['page'] for x in test)
    x_test = hstack((x_test_sentence,x_test_genre))

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
