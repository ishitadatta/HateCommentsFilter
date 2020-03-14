import pandas as pd
import numpy as np

import warnings

from django.core.cache import cache

from os import path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from .utils import clean_and_tokenize_tweets


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class Classifier:

    model_filename = '/code/hate_speech_AI/model-cache/cache.pkl'
    model_cache_key = 'model_cache'
    model_rel_path = "hate_speech_AI/model-cache/cache.pkl"

    vectorizer_filename = '/code/hate_speech_AI/vectorizer-cache/cache.pkl'
    vectorizer_cache_key = 'vectorizer_cache'
    vectorizer_rel_path = "hate_speech_AI/vectorizer-cache/cache.pkl"

    score_cache_key = 'score_cache'

    def __init__(self):

        # TF-IDF feature matrix
        tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer = tfidf_vectorizer
        self.lreg = LogisticRegression()

#intializing the class
classifier = Classifier()

# We first train the model at the creation of the model
train = pd.read_csv('/code/hate_speech_AI/bad-words.csv')

#The posts contain lots of twitter handles (@user), which we will remove from the data as they don’t convey much information.
#def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

#create a new column tidy_tweet which will contain the cleaned and processed tweets
def clean_and_tokenize_tweets(data):
    # Data has to be a dict containing tweets
    data['tidy_tweet'] = np.vectorize(remove_pattern)(data['tweet'], "@[\w]*")

# remove special characters, numbers, punctuations
    data['tidy_tweet'] = data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
# remove words with 3 or less character, presumibily not useful
    data['tidy_tweet'] = data['tidy_tweet'].apply(
        lambda x: ' '.join([w for w in x.split() if len(w) > 3])
    )
    # Tokenize the words for it to use
    tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split()

    from stemming.porter2 import stem

    # Stem the words. (Stemming is a rule-based process of stripping
    # the suffixes (“ing”, “ly”, “es”, “s” etc) from a word)

    tokenized_tweet = tokenized_tweet.apply(
        lambda x: [stem(i) for i in x]
    )
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

#train the model
def train_model(self):

        # We first train the model at the creation of the model
        train = pd.read_csv('/code/hate_speech_AI/train.csv')
        # Remove all the unncessary parts in the

        train_data = clean_and_tokenize_tweets(train)
        tfidf = self.tfidf_vectorizer.fit_transform(train_data['tidy_tweet'])

        # Building model using TF-IDF features

        # splitting data into training and validation set
        xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(
            tfidf, train['label'], random_state=42, test_size=0.25,
        )

        self.lreg.fit(xtrain_tfidf, ytrain)
        if not (path.exists(self.model_filename)):
            model_file = open(self.model_filename, 'w+')
            model_file.close()
        if not (path.exists(self.vectorizer_filename)):
            vectorizer_file = open(self.vectorizer_filename, 'w+')
            vectorizer_file.close()
        joblib.dump(self.lreg, self.model_filename)
        joblib.dump(self.tfidf_vectorizer, self.vectorizer_filename)

         # Get the score of the model
        prediction = self.lreg.predict_proba(xvalid_tfidf)
        prediction_int = prediction[:, 1] >= 0.75
        prediction_int = prediction_int.astype(np.int)

        cache.set(self.model_cache_key, self.lreg, None)
        cache.set(self.vectorizer_cache_key, self.tfidf_vectorizer, None)
        cache.set(self.score_cache_key, f1_score(yvalid, prediction_int), None)

 def get_score(self):
        score = cache.get(self.score_cache_key)
        if score:
            return score
        return 'No score in cache'

#prediction
 def predict_single_tweet(self, tweet):
        # return 1 if offensive, 0 if not
        if type(tweet) != str:
            return
        model = cache.get(self.model_cache_key)
        vectorizer = cache.get(self.vectorizer_cache_key)
        if model is None:
            model_path = path.realpath(self.model_rel_path)
            model = joblib.load(model_path)

            # save in django memory cache

            cache.set(self.model_cache_key, model, None)

        if vectorizer is None:
            vectorizer_path = path.realpath(self.vectorizer_rel_path)
            vectorizer = joblib.load(vectorizer_path)

            # save in django memory cache

            cache.set(self.vectorizer_cache_key, vectorizer, None)
        tweet_to_predict = vectorizer.transform([tweet])
        return model.predict(tweet_to_predict)[0]