import gensim.models.keyedvectors as word2vec #need to use due to depreceated model
from nltk.tokenize import RegexpTokenizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers import LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve,  roc_auc_score, classification_report
import logging
logging.basicConfig(format='%(asctime)s : %(levelname) s : %(message)s', level=logging.INFO)

#Set random seed
np.random.seed(24)
#read CSV file containing tweets and labels, using Pandas , to get a dataframe
tweetsData = pd.read_csv('data.csv')
tweetsData.head()
#Dividing the dataset into features and lables
tweets = tweetsData['SentimentText']
labels = tweetsData['Sentiment']
print(tweets)
print(labels)
#check the distribution of lebels
labels_count = labels.value_counts()
labels_count.plot(kind="bar")
print(labels.value_counts())

#Looks like the distribution is even

#Lower and split the dialog
#and use regular expression to keep only letters we will use nltk Regular expression package
tkr = RegexpTokenizer('[a-zA-Z@]+')
tweets_split = []
for i, line in enumerate(tweets):
    #print(line)
    tweet = str(line).lower().split()
    tweet = tkr.tokenize(str(tweet))
    tweets_split.append(tweet)

print(tweets_split[1])

'''
Use pretrained Word2Vec model from google but trim the word list to 50,0000 compared to 300,000 in the original
Google pretrained model
'''
w2vModel = word2vec.KeyedVectors.load_word2vec_format('D:/twittersentiment1/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=50000)
#Convert words to integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_split)
X = tokenizer.texts_to_sequences(tweets_split)
#lenght of tweet to consider
maxlentweet = 10
#add padding
X = pad_sequences(X, maxlen=maxlentweet)
print(X.shape)
#create a embedding layer using Google pre triained word2vec (50000 words)
embedding_layer = Embedding(input_dim=w2vModel.syn0.shape[0], output_dim=w2vModel.syn0.shape[1], weights=[w2vModel.syn0],
                            input_length=X.shape[1])
#create model
lstm_out = 80
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(units=lstm_out))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, labels, test_size= 0.1, random_state = 24)
#fit model
batch_size = 1
model.fit(X_train, Y_train, epochs=10, verbose=1, batch_size=batch_size)
#analyze the results
score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size=batch_size)
print("testdata......",X_test)
y_pred = model.predict(X_test)
print(y_pred)
