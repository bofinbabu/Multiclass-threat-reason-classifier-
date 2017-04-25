# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

embeddings_index = {}

if (os.path.isfile('embeddings_index.pickle')) == False :
    print 'pickle not found, creating and loading...'
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    #print embeddings_index
    import pickle
    with open('embeddings_index.pickle', 'wb') as handle:
        pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


print 'embdeddings file exists, loading...'
import pickle
with open('embeddings_index.pickle', 'rb') as handle:
    embeddings_index = pickle.load(handle)
print 'loaded'


print('Total %s word vectors.' % len(embeddings_index))
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


rev_text = ["Clinicopathological features of elevated lesions of the duodenal bulb. We present here  our findings on patients with an elevated lesion of the duodenal bulb. All these patients were treated in our clinics between the years 1984 and 1988. These lesions were present in 36 of 8 802 patients who underwent upper gastrointestinal pan-endoscopy. Two patients had a duodenal carcinoma  2 an adenoma  and 1 a Brunner s gland adenoma. There were 15 with a hyperplastic polyp  3 with a heterogenic gastric mucosa  3 with Brunner s gland hyperplasia  6 with duodenitis  and 4 with regenerative mucos a. Among these 36 lesions  only 69   25 lesions  were evident on the upper gastrointestinal X-ray series. Adenoma and Brunner s gland adenoma    were of a pedunculated form of the gross type and had an irregular surface mucosa. Both duodenal carcinomas were detected by endoscopic biopsy and   were resected. Histologically  these lesions were limited to the submucosal layer and were of the non","Hello I am Mayank"]
rev_text = np.asarray(rev_text)
print 'rev_text shape si ',rev_text.shape

from nltk import tokenize
reviews = []
texts = []
for idx in range(rev_text.shape[0]):
    text = BeautifulSoup(rev_text[idx],"lxml")
    text = clean_str(text.get_text().encode('ascii','ignore'))
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    reviews.append(sentences)


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)


data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1

word_index = tokenizer.word_index
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
print 'shape of data is  ',data.shape

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(50))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(50, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(50))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(3, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',    metrics=['acc'])
#model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch= 1, batch_size=50)

from keras.models import load_model
from keras.models import model_from_json

model.load_weights('hat_model_4_classes.h5')
print("model weights loaded compiled, now compiling - Hierachical attention network")

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print 'model compiled succesfully  :) '

predictions = model.predict(data);
print 'probablities for classes are: ',predictions
print 'classes index are (+1): '
for prediction in predictions:
    print np.argmax(prediction)
