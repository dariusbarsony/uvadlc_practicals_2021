###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

import math
import random
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(lstm_hidden_dim))

        # candidate values
        self.W_gx = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(lstm_hidden_dim)) 

        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(embedding_size, lstm_hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(lstm_hidden_dim))
        
        self.h_t = None
        self.c_t = None

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        sigma = 1.0 / math.sqrt(self.hidden_dim)

        for w in self.parameters():
            w.data.uniform_(-sigma, sigma)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds, states=None):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        il = embeds.shape[0]
        bs = embeds.shape[1]

        hs = []

        device = self.W_ix.device

        if states:
            self.h_t, self.c_t = states
        else:
            self.h_t = torch.zeros(bs, self.hidden_dim, device=device)
            self.c_t = torch.zeros(bs, self.hidden_dim, device=device)

        for t in range(il):
            x_t = embeds[t]

            i_t = torch.sigmoid(x_t @ self.W_ix + self.h_t @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_fx + self.h_t @ self.W_fh + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_ox + self.h_t @ self.W_oh + self.b_o)

            g_t = torch.tanh(x_t @ self.W_gx + self.h_t @ self.W_gh + self.b_g)

            self.c_t = g_t * i_t + self.c_t * f_t
            self.h_t = torch.tanh(self.c_t) * o_t

            hs.append(self.h_t.unsqueeze(0))
        
        hs = torch.cat(hs, dim=0)

        return hs

        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # batch size, vocabulary size
        self.batch_size = args.batch_size
        self.vocabulary_size = args.vocabulary_size

        # embedding size, hidden size
        self.embedding_size = args.embedding_size 
        self.hidden_size = args.lstm_hidden_dim
        
        # store device
        self.device = args.device

        # character translations
        self._ix_to_char = args._ix_to_char
        self._char_to_ix = args._char_to_ix

        # create model, specify embedding, introduce linear output layer
        self.model = LSTM(self.hidden_size, self.embedding_size)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.fc1 = nn.Linear(self.hidden_size, self.vocabulary_size)
        
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
    
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        embedded_x = self.embedding(x)
        x_ = self.model(embedded_x)
        x_ = self.fc1(x_)

        return x_

        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=4, sample_length=30, temperature=5):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        start = random.randint(0, self.vocabulary_size + 1)
        sample_list = [start]

        for _ in range(sample_length):

            # get last character, add dimension, push to device, put into LongTensor
            input_ = torch.LongTensor([sample_list[-1]]).unsqueeze(-1).to(self.device)
            out_ = self(input_)

            if temperature:
                preds = torch.softmax(temperature * out_, dim=2)
                char = torch.multinomial(preds[-1], num_samples=1, replacement=True)
            else:
                preds = torch.softmax(out_, dim=2)
                char = torch.argmax(preds[-1])
            sample_list.append(int(char))

        # store samples in model
        self.sample_list = sample_list

        #######################
        # END OF YOUR CODE    #
        #######################
