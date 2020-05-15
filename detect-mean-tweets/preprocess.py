import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.stem.porter import *
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

#Data Preprocessing 
def pre_process(train, test):



    #Visualize how data looks like
    #print(train.head())

    #Visualize only offensive tweets
    #print(train[train['label'] == 1])

    ## Data Cleaning ##

    #Combine training and test datasets, to make everything easier!

    combi = train.append(test, ignore_index=True)

    #Remove Twitter handles

    combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
    #print(combi['tidy_tweet'])

    # Remove special characters, numbers, punctuation

    combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " " )
    #print(combi['tidy_tweet'])

    # Removing short words

    combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    ## Tokenization ##

    tokenized_tweet = combi['tidy_tweet'].apply(lambda x:x.split())

    ## Stemming ##

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    ## Pushing changes ##

    for i in range (len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    combi["tidy_tweet"] = tokenized_tweet

    return combi