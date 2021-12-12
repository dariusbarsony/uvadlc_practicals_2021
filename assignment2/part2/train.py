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

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)

    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)

    # add vocabulary size                      
    args.vocabulary_size = dataset.vocabulary_size

    # translational dictionaries   
    args._ix_to_char = dataset._ix_to_char
    args._char_to_ix = dataset._char_to_ix

    # Create model, init optimizer, init loss module
    LSTM = TextGenerationModel(args)

    optimizer = optim.Adam(LSTM.model.parameters(), lr=args.lr)
    loss_module = nn.CrossEntropyLoss()

    # push model to device and set to train mode
    LSTM.to(args.device)
    LSTM.train()
    
    # initialise plotting lists
    accuracies_train = []
    losses_train = []
    
    results = {}

    # per epoch
    for e in range(args.num_epochs):
        acc_train = 0.0
        loss_train = 0.0

        # per batch 
        for data_inputs, data_labels in data_loader:

            # push labels to device
            data_inputs = data_inputs.to(args.device)
            data_labels = data_labels.to(args.device)

            # propagate inputs through network
            preds = LSTM(data_inputs)
            
            # calculate cross-entropy loss per timestep
            loss = 0.0

            for i in range(args.input_seq_length):
                loss += loss_module(preds[i, :, :], data_labels[i])

            # average loss and compute accuracy
            loss /= args.input_seq_length
            acc = (data_labels == preds.argmax(dim=-1)).float().mean()
            
            loss_train += loss
            acc_train += acc

            # propagate loss backwards            
            loss.backward()

            # apply gradient clipping, optimizer step
            nn.utils.clip_grad_norm_(LSTM.model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
        
        # loss/accuracy per epoch
        losses_train += [int(loss_train/args.batch_size)]
        accuracies_train += [int(acc_train/args.batch_size)]

    results['losses'] = losses_train
    results['accuracies'] = accuracies_train

    # store model
    torch.save(LSTM.state_dict(), 'LSTM.pt')

    # write results to csv file
    if args.save_results:
        # store results in dictionary
        with open('results_lstm.csv', 'w+') as f:
            for key in results.keys():
                f.write("%s, %s\n" %(key, results[key]))

    # sample
    if args.sample:
        LSTM.sample()
        print(dataset.convert_to_string(LSTM.sample_list))

    #######################
    # END OF YOUR CODE    #
    #######################

'''
Helper function for plotting the LSTM losses and accuracies. 
'''
def plotModel(results_path, metric, epochs):
    
    # read in resnet dataset and create df
    # data = pd.read_csv(results_path)
    # df = pd.DataFrame(data)
    
    fig,ax = plt.subplots()

    # hardcoded results, because format of 
    # results file was not what I expected. It included the device 
    # and there was no more time left to train
    
    loss = [21.9438, 21.9170, 21.9074, 21.8990, 21.9030, 21.8949, 21.8936, 21.8963, 21.8870, 21.8875, 21.8784, 21.8805, 21.8780, 21.8722, 21.8682, 21.8670, 21.8626, 21.8678, 21.8611, 21.8593]
    accuracy = [1.1454, 1.1292, 1.1300, 1.1305, 1.1292, 1.1294, 1.1307, 1.1299, 1.1303, 1.1294, 1.1292, 1.1299, 1.1297, 1.1305, 1.1298, 1.1296, 1.1302, 1.1297, 1.1282, 1.1300]

    ax.plot(range(1, epochs+1), accuracy, label='lstm accuracies')

    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    ax.legend(loc='best')

    fig.savefig('lstm_accuracy.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    # added flags
    parser.add_argument('--plot', type=bool, default=False, help='Flag for plotting.')
    parser.add_argument('--save_results', type=bool, default=True, help='Argument for saving results.')
    parser.add_argument('--sample', type=bool, default=False, help='Argument for whether to sample or not.')


    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    train(args)

    if args.plot:
        plotModel('', '', 20)

