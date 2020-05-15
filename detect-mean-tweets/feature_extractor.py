from preprocess import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Read training and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

combi = pre_process(train, test)

def bow_feature():
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
    
    train_bow = bow[:31962, :]
    test_bow = bow[31962:, :]

    return train_bow, test_bow

def tfidf_feature():
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

    train_tfidf = tfidf[:31962, :]
    test_tfidf = tfidf[31962:, :]

    return train_tfidf, test_tfidf