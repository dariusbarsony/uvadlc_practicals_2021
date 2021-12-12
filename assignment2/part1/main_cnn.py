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
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import pickle as pl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set


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


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir)

    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=True, drop_last=False)

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    loss_module = nn.CrossEntropyLoss()

    # set model to train mode
    model.to(device)
    best_acc = 0.0
    
    # Test best model on validation and test set
    for e in range(epochs):
        # train_acc = 0.0
        valid_acc = 0.0
        model.train()

        for data_inputs, data_labels in train_dataloader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            loss = loss_module(preds, data_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # multistep LR
        scheduler.step()
        
        # calculate accuracy
        valid_acc = evaluate_model(model, val_dataloader, device)

        # find best model
        if best_acc < valid_acc:
            torch.save(model.state_dict(), checkpoint_name)

        print('VALID ACC: %.4f'%(valid_acc))

    # Load best model and return it.
    try:
        model.load_state_dict(torch.load(checkpoint_name))
    except:
        print('no saved model found')

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    model.to(device)

    acc = 0.0
    for data_inputs, data_labels in data_loader:
        data_inputs = data_inputs.to(device)
        data_labels = data_labels.to(device)

        preds = model(data_inputs)
        preds = preds.squeeze(dim=1)

        acc += (preds.argmax(dim=-1) == data_labels).float().mean()
    
    # average over batches
    accuracy = acc/len(data_loader)

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy

def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    set_seed(seed)
    test_results = {}
    
    # add 'vanilla' test result to overall results
    test_dataset = get_test_set(data_dir)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
    test_results[('vanilla', 'x')] = float(evaluate_model(model, test_dataloader, device))

    augmentations = {'gaussian noise':gaussian_noise_transform, 'gaussian blur':gaussian_blur_transform, 'contrast':contrast_transform, 'jpeg':jpeg_transform}
    severities = range(1, 6)

    for aug in augmentations.keys():
        for s in severities:
            corruption = augmentations[aug](s)
            test_dataset = get_test_set(data_dir, corruption)

            test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
            test_results[('{}'.format(aug), s)] = float(evaluate_model(model, test_dataloader, device))

    return test_results

    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed, train=True, test=True, dump=True, show=False):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # initialise device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)
    
    # get checkpoint name
    checkpoint_name = 'models/{}.pt'.format(model_name)
    
    # check whether existing model exists
    try:
        model = get_model(model_name)
        model.load_state_dict(torch.load(checkpoint_name))
    except:
        model = get_model(model_name)
    
    # train and test model
    if train:
        print('training of {} started'.format(model_name))
        model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device)
        print('training ended')
    if test:
        print('testing started')
        test_results = test_model(model, batch_size, data_dir, device, seed)

        print('testing ended')
        print('test_results', test_results)

    # dump results in a file
    if dump:
        import csv

        f = open('results_{}.csv'.format(model_name), 'w+')

        header = ['corruption', 'severity', 'test result']

        writer = csv.writer(f)
        writer.writerow(header)
        
        for key in test_results.keys():

            row = [key[0], key[1], test_results[key]]

            # write a row to the csv file
            writer.writerow(row)

        # close the file
        f.close()

    if show:
        results = open("results.pkl", "rb")
        pl.load(results)
        print(results)
    
    #######################
    # END OF YOUR CODE    #
    #######################

def plotModel(resnet_results, corruptions):
    
    # read in resnet dataset and create df
    data_resnet = pd.read_csv(resnet_results)
    df_resnet = pd.DataFrame(data_resnet)
    
    fig,ax = plt.subplots()
    
    # loop over corruptions
    for i in corruptions:
        ax.plot(df_resnet[df_resnet.corruption==i].severity,df_resnet[df_resnet.corruption==i].test_result, label=i)

        ax.set_xlabel("severity")
        ax.set_ylabel("accuracy")
        ax.legend(loc='best')
    fig.savefig('plots/resnet18.png')

def plotCE(resnet_results, other_results, corruptions, models, metric='RCE'):
    
    # read in resnet dataset and create df
    data_resnet = pd.read_csv(resnet_results)
    df_resnet = pd.DataFrame(data_resnet)
    
    # initialise dataholder
    CE = {'corruptions':corruptions}
    
    # loop over models
    for k in range(len(other_results)):
        # get results model
        data_other = pd.read_csv(other_results[k])
        df_other = pd.DataFrame(data_other)
        
        # create model result list
        CE[models[k]] = []
        
        # loop over corruptions
        for i in range(len(corruptions)):
            # get corruption results for model
            df = df_resnet[df_resnet["corruption"] == corruptions[i]]
            df2 = df_other[df_other["corruption"] == corruptions[i]]
            
            # calculate error rate
            error_rate = 1  -  df['test_result']
            error_rate_other = 1  -  df2['test result']
            
            # if RCE
            if metric == 'RCE':
                
                # get clean column
                clean_resnet = df_resnet[df_resnet["corruption"] == 'vanilla']
                clean_other = df_other[df_other["corruption"] == 'vanilla']
                
                # calculate clean error rate
                ecr = 1  -  clean_resnet['test_result']
                eco = 1  -  clean_other['test result']
                
                # subtract clean from error rate
                error_rate = error_rate.subtract(float(ecr)) 
                error_rate_other = error_rate_other.subtract(float(eco))
                            
            # sum
            resnet18 = error_rate.sum(axis=0)
            other = error_rate_other.sum(axis=0)
                        
            # put metric in list
            CE[models[k]] += [other/resnet18]
    
    final = pd.DataFrame(CE)

    y = [i for i in models]
    fig = final.plot(x="corruptions", y=y, kind="bar")

    plt.tight_layout()
    
    fig.get_figure().savefig('plots/{}.png'.format(metric))

if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')
    
    # additional parse arguments
    parser.add_argument('--plot', default=False, type=bool,
                        help='Whether to plot or not')
    parser.add_argument('--metric', default='CE', type=str,
                        help='metric to use for plotting')

    args = parser.parse_args()
    kwargs = vars(args) 
    main(**kwargs)
    
    if args.plot:
        PATH_RESNET18 = 'results/results_resnet18.csv'
        PATH_RESNET34 = 'results/results_resnet34.csv'
        PATH_VGG11 = 'results/results_vgg11.csv'
        PATH_VGG11_BN = 'results/results_vgg11_bn.csv'
        PATH_DENSENET121 = 'results/results_densenet121.csv'

        augmentations = {'gaussian noise':1, 'gaussian blur':2, 'contrast':3, 'jpeg':4}
        modelNames = ['resnet34', 'vgg11', 'densenet121']

        corruptions = list(augmentations.keys())

        # left out because model not converging
        # PATH_VGG11_BN,  'vgg11_bn'

        plotCE(PATH_RESNET18, [PATH_RESNET34, PATH_VGG11, PATH_DENSENET121], corruptions, modelNames, metric=args.metric)
        plotModel(PATH_RESNET18, corruptions)
