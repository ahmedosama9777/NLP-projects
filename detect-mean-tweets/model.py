from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from feature_extractor import *

def bow_model():
    train_bow, test_bow = bow_feature()

    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain)

    prediction = lreg.predict_proba(xvalid_bow)
    prediction_int = prediction[:,1] >= 0.3
    prediction_int = prediction_int.astype(np.int)

    print(f1_score(yvalid, prediction_int))
    return f1_score(yvalid, prediction_int)

def tfidf_model():
    train_tfidf, test_tfidf = tfidf_feature()

    xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.3)

    lreg = LogisticRegression()
    lreg.fit(xtrain_tfidf, ytrain)

    prediction = lreg.predict_proba(xvalid_tfidf)
    prediction_int = prediction[:,1] >= 0.3
    prediction_int = prediction_int.astype(np.int)

    print(f1_score(yvalid, prediction_int))
    return f1_score(yvalid, prediction_int)
