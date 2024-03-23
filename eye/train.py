import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_loader import *
import argparse
from models import * 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import optuna

def train(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    """This function trains the model for a specified number of epochs.

    Args:
        train_loader: the data loader for the training data
        model: the model to be trained
        optimizer: the optimizer
        num_epochs: the number of epochs to train the model
    """
    # Initialize list to store losses for each epoch
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

        for i, (inputs, labels, input_lengths) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, input_lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            predicted = torch.max(outputs, 1)[1]  # Use torch.max to get the index of the max log-probability.
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct / total_samples
        print(f'Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Append the loss for this epoch to the list of losses
        train_losses.append(epoch_loss)

        val_epoch_loss, val_epoch_acc = evaluate(model, val_loader, criterion)

        print(f'Validation Epoch: [{epoch+1}/{num_epochs}], Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_acc:.4f}')

        val_losses.append(val_epoch_loss)

        # save model if validation loss has decreased
        if val_epoch_loss <= best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {val_epoch_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), 'model.pt')
            best_val_loss = val_epoch_loss
            best_val_acc = val_epoch_acc

    print(f'Best validation loss: {best_val_loss:.6f}, Best validation accuracy: {best_val_acc:.6f}')

    # After training is complete, plot the losses and save the figure locally
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    #Save figure locally
    plt.savefig('training_validation_loss.png')
    plt.show()

def evaluate(model, test_loader, criterion, print_info=False):
    """This function evaluates the model on the test set.
    Args:
        model: the model to be evaluated
        test_loader: the data loader for the test data
        criterion: the criterion to compute the loss
        print_info: whether to print the loss, the accuracy and the misclassifications per class. Default is False.
    """
    model.eval()
    test_correct = 0
    test_total = 0
    misclassifications = {0: 0, 1: 0}  # Initialize misclassifications per class
    test_loss = 0

    with torch.no_grad():
        for inputs, labels, input_lengths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            test_outputs = model(inputs, input_lengths)
            labels = labels.long()  # Convert labels to long

            test_predicted = torch.max(test_outputs, 1)[1]  # get the index of the max log-probability
            test_correct += (test_predicted == labels).sum().item()
            test_total += labels.size(0)

            # Track misclassifications per class
            for i in range(labels.size(0)):
                if test_predicted[i] != labels[i]:
                    misclassifications[int(labels[i].item())] += 1

            #Compute the loss and accumulate it
            loss = criterion(test_outputs, labels)
            test_loss += loss.item() * inputs.size(0)

    #Calculate the average test loss over all batches
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    #Print the loss, the accuracy and the misclassifications per class
    if print_info:
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        for class_label, misclassified_count in misclassifications.items():
            print(f'Misclassifications for class {class_label}: {misclassified_count}')
    
    return test_loss, test_accuracy


def objective(trial, train_loader, val_loader, input_size, device):
    """Function to be optimized by Optuna.
    Args:
        trial: the current trial
        train_loader: the data loader for the training data
        val_loader: the data loader for the validation data
        input_size: the size of the input
        device: the device to be used for training
        """
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [8, 15, 20, 31])
    #num_epochs = trial.suggest_int('num_epochs', 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    hidden_size =  64 #trial.suggest_categorical('hidden_size', [32, 64, 100])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    model = trial.suggest_categorical('model', ['GRU', 'LSTM', "ModifiedGRU", "ModifiedLSTM"])

    # Create the model
    if model == 'GRU':
        model = GRUModel(input_size, hidden_size, num_layers).to(device)
    elif model == 'LSTM':
        model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    elif model == 'ModifiedLSTM':
        model = ModifiedLSTMModel(input_size, hidden_size, num_layers, dropout_prob=dropout).to(device)
    elif model == 'ModifiedGRU':
        model = ModifiedGRUModel(input_size, hidden_size, num_layers, dropout_prob=dropout).to(device)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train and evaluate the model, then return the validation loss
    train(train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs)

    # Load the best model
    model.load_state_dict(torch.load('model.pt'))

    val_loss, val_acc = evaluate(model, val_loader, criterion)

    return val_loss

def hyperparameter_tuning(train_loader, val_loader, test_loader, input_size, device, num_trials=50):
    """Function to tune the hyperparameters of the model using Optuna.
    Args:
        num_trials (int): Number of hyperparameter tuning trials to run.
        train_loader: the data loader for the training data
        val_loader: the data loader for the validation data
        input_size: the size of the input
        device: the device to be used for training
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader, input_size, device), n_trials=num_trials)

    print('Best trial:')
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Create a new model with the best hyperparameters
    if trial.params['model'] == 'GRU':
        best_model = GRUModel(input_size, 64, trial.params['num_layers']).to(device)
    elif trial.params['model'] == 'LSTM':
        best_model = LSTMModel(input_size, 64, trial.params['num_layers']).to(device)
    elif trial.params['model'] == 'ModifiedLSTM':
        best_model = ModifiedLSTMModel(input_size, 64, trial.params['num_layers'], dropout_prob=trial.params['dropout']).to(device)
    elif trial.params['model'] == 'ModifiedGRU':
        best_model = ModifiedGRUModel(input_size, 64, trial.params['num_layers'], dropout_prob=trial.params['dropout']).to(device)
    
    # Create the optimizer with the best learning rate and weight decay
    optimizer = optim.Adam(best_model.parameters(), lr=trial.params['learning_rate'], weight_decay=trial.params['weight_decay'])

    # Train the model with the best number of epochs
    train(train_loader, val_loader, best_model, criterion, optimizer, num_epochs=25)

    # Load the best model after training
    best_model.load_state_dict(torch.load('model.pt'))

    # Evaluate the model on the test set
    test_loss, test_acc = evaluate(best_model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='number of hidden units')
    parser.add_argument('--num_layers', type=int, default=5, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    #Add an argument for the model lst, gru, modifiedlstm, modifiedgru, resnet_lstm
    parser.add_argument('--model', type=str, default='gru', help='model type', choices=['lstm', 'gru', 
                                                                                        'modifiedlstm', 'modifiedgru', 
                                                                                        'gru_ln'])
    parser.add_argument('--input_size', type=int, default=8, help='number of input features')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--tune_hyperparameters', action='store_true', help='tune hyperparameters')
    args = parser.parse_args()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the hyperparameters
    input_size = args.input_size  # Number of input features (8 gaze features)
    num_classes = 2  # Number of unique classes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    model_type = args.model
    weight_decay = args.weight_decay
    tune_hyperparameters = args.tune_hyperparameters

    if model_type == 'lstm':
        model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    elif model_type == 'gru':
        model = GRUModel(input_size, hidden_size, num_layers).to(device)
    elif model_type == 'modifiedlstm':
        model = ModifiedLSTMModel(input_size, hidden_size, num_layers, dropout_prob=dropout).to(device)
    elif model_type == 'modifiedgru':
        model = ModifiedGRUModel(input_size, hidden_size, num_layers, dropout_prob=dropout).to(device)
    elif model_type == 'gru_ln':
        model = GRUModel_LN(input_size, hidden_size, num_layers).to(device)
   
    
    # Create data loaders for training and testing
    train_loader, test_loader, val_loader = get_loader(batch_size=batch_size, normalize=True)   

    # Define the loss function and optimizer
    class_weights = class_weights(train_loader)
    print(class_weights.size())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) #nn.BCELoss(weight = class_weights.to(device)) #

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    train(train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs)

    # Load the best model
    model.load_state_dict(torch.load('model.pt'))

    # Evaluate the model on test data
    evaluate(model, test_loader, criterion, print_info=True)

    # Create the study and optimize
    if tune_hyperparameters:
        hyperparameter_tuning(num_trials=10, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, input_size=input_size, device=device)