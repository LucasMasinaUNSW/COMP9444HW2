#!/usr/bin/env python3
"""
z5204754

Question:
- chose to use LSTM
    - thought long term memory was useful
    - given that the inputs are sentences, the context of a word within a sentence greatly affects
        the meaning of it and its more about finding the 'tone' of the review
    - more complicated but allowed for more refinement

- ignore padded inputs
- preprocessing:
    - removing punctuation
    - removing special characters
- dropout
- remove common and uncommon words i.e. stop words
- fiddling with num of hidden nodes, learning rate, momentum
- attempted to use a regression model using MSELoss instead of classification with CrossEntropy
    and noticed that the network was less likely to make a prediction further away (i.e. 3 or 4 stars away),
    it was also less likely to correctly predict and more likely to predict one star away, reducing the overall
    weighted score.

"""


"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

# TODO check if can use the below packages
import re
import string
# import spacy
import nltk


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove punctuation, special characters
    input = " ".join(sample)
    text = re.sub(r"[^\x00-\x7F]+", " ", input)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    result = nopunct.split(" ")
    result = list(filter(lambda x: x != '', result))

    # print(result)
    return result


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    # print("batch: ", batch)
    # print("vocab: ", vocab.freqs)

    # Remove infrequent words from batch
    vocabCount = vocab.freqs
    vocabITOS = vocab.itos

    for i, x in enumerate(batch):
        for j, y in enumerate(x):
            if vocabCount[vocabITOS[y]] < 3:
                x[j] = -1
        # batch[i] = list(filter(lambda a: (vocabCount[vocabITOS[a]] > 2), x))

    # print("new batch: ", batch)
    return batch


# spacy.load('en_core_web_sm')
# stopWords = spacy.lang.en.stop_words.STOP_WORDS
nltk.download('stopwords')
# stopWords = nltk.corpus.stopwords.words('english')[:77]
# print("stopWords: ", stopWords)
stopWords = {}

wordVectorDimension = 200
wordVectors = GloVe(name='6B', dim=wordVectorDimension)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################


def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    out = datasetLabel.long() - 1   # Necessary since for each label: 0 <= label < n_classes and label needs to be of type long (requirement of criterion)
    return out


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    out = (torch.argmax(netOutput, dim=1) + 1).float()  # Gets the index of the highest probability, adds 1 then converts to float
    # out = torch.round((netOutput + 1).float())
    return out

###########################################################################
################### The following determines the model ####################
###########################################################################


class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()

        hidden_dim = 200
        num_layers = 1
        out_dim = 5
        drop_rate = 0.2

        self.lstm = tnn.LSTM(input_size=wordVectorDimension, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = tnn.Linear(in_features=num_layers*hidden_dim, out_features=out_dim)
        self.dropout = tnn.Dropout(drop_rate)

    def forward(self, input, length):
        embedded = self.dropout(input)
        embedded = tnn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)    # Ignores padded inputs
        output, (hidden, cell) = self.lstm(embedded)

        # hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = hidden[-1]

        outputs = self.linear(hidden)

        return outputs


# TODO Implement custom loss function probably
# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#
#     def forward(self, output, target):
#         pass      # Maybe implement cost = -T.mean(target * T.log(y_vals)+ (1.- target) * T.log(1. - y_vals)) where T is tnn


net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
# lossFunc = loss()     # TODO use this for custom loss function
lossFunc = tnn.CrossEntropyLoss()     # shouldn't use with log_softmax() apparently since its already used within
# lossFunc = tnn.MSELoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.08, momentum=0.8)
