#!/usr/bin/env python3
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


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # TODO ignore common words (e.g. the, a etc.), maybe also infrequent words
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

stopWords = {}
wordVectorDimension = 50
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

        hidden_dim = 100
        num_layers = 2
        out_dim = 5
        self.lstm = tnn.LSTM(input_size=wordVectorDimension, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = tnn.Linear(in_features=2*hidden_dim, out_features=out_dim)
        # self.act = tnn.Sigmoid()    # maybe Softmax?

    def forward(self, input, length):
        # input:  [batch_size (e.g. 32), num_vectors, vector_dim=wordVectorDimension (e.g. 50)] - embeddings
        # length: [batch_size (e.g. 32)]                                                        - max lengths of reviews

        embedded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)    # Ignores padded inputs

        output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        dense_outputs = self.linear(hidden)                                                 # [32, 5]

        # outputs = self.act(dense_outputs)
        outputs = F.log_softmax(dense_outputs, dim=1)

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
# lossFunc = tnn.CrossEntropyLoss()     # shouldn't use with log_softmax() apparently since its already used within
lossFunc = tnn.NLLLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.1)
