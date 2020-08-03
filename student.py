#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

z5204754, z5149715

How the program works and decisions made:
Firstly, punctuation, special characters and numbers are removed from each review in the pre-processing step.
This is done since it was decided that these do not offer much meaningful value as they are often specific to the
product being reviewed such as the quantity of the product or specific to the reviewer's writing style and therefore
harmful for predictions.

Next, in the post-processing stage, for each batch, words that occur less than 3 times in the training set of reviews
are given a default value. This is so the network won't take them into consideration during training as they are often
typos or other infrequent words that do not benefit the model.

No stop words were added since during experimentation they always decreased performance. A dimension of 200 for GloVe
vectors proved to be the best after testing. Attempts with 300, 100 and 50 performed significantly worse. The labels
were converted to integers with range of [0, 4] in convertLabels, since the loss function requires it. The network
returns probabilities for each rating class for every review and so the class with the highest probability is selected
and then converted to floats with range of [1.0, 5.0] in convertNetOutput.

The actual model used was an LSTM using CrossEntropy as the loss function. This was chosen due to its ability
to mimic long term memory which would help learn to 'understand' the context of words in a sentence affecting their
meaning and potentially 'understand' the overall tone of the review. Other models were also tested including RNN and
CNN, but LSTM produced the best results by a significant amount. Another consideration was switching to a regression
model rather than classification and using Mean Squared Error for the loss function. While this did reduce the number of
results off by 2, 3 or 4 stars as it punishes predictions further away from target, there was a lower overall score
since the number  of results off by 1 star increased and the correct predictions rate decreased.

For each training pass through, a dropout layer is used to add some randomness and prevent overfitting, padded inputs
are ignored and then a loss value is calculated using CrossEntropyLoss. This loss function applies Softmax and then
calculates the error of the predicted result compared with the correct review ratings with Negative Likelihood Loss,
punishing incorrect predictions proportionate to the confidence of the prediction. Schocastic Gradient Descent is used
to optimise the network based on the prediction error. Although a slower optimiser, it proved to return the best results
compared to using ADAM. Finally, the solution was refined by testing results with different hyper parameters such as
number of hidden nodes in the LSTM layer, learning rate and momentum.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
import re
import string
from torchtext.vocab import GloVe


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # Remove punctuation, special characters and numbers
    input = " ".join(sample)
    text = re.sub(r"[^\x00-\x7F]+", " ", input)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    result = nopunct.split(" ")
    result = list(filter(lambda x: x != '', result))
    return result


def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    # Remove infrequent words from batch
    vocabCount = vocab.freqs
    vocabITOS = vocab.itos
    for sentence in batch:
        for j, word in enumerate(sentence):
            if vocabCount[vocabITOS[word]] < 3:
                sentence[j] = 0
    return batch


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
    # Convert to integers with range of [0, 4]
    out = datasetLabel.long() - 1
    return out


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # Find most likely class
    bestClass = torch.argmax(netOutput, dim=1)

    # Convert to floats with range [1.0, 5.0]
    out = (bestClass + 1).float()
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

        # Model parameters
        hidden_dim = 200
        num_layers = 1
        out_dim = 5
        drop_rate = 0.2

        # Initialising model
        self.lstm = tnn.LSTM(input_size=wordVectorDimension, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = tnn.Linear(in_features=num_layers * hidden_dim, out_features=out_dim)
        self.dropout = tnn.Dropout(drop_rate)

    def forward(self, input, length):
        # Randomise occasionally
        embedded = self.dropout(input)

        # Ignore padded inputs
        packed_input = tnn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=True)

        # Run through LSTM
        output, (hidden, cell) = self.lstm(packed_input)

        # Run through linear layer
        outputs = self.linear(hidden[-1])
        return outputs


net = network()

# Executes Softmax and then uses NLLLoss to calculate error margin
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

# Hyper parameters
trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.08, momentum=0.8)
