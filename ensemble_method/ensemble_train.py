
import numpy as np
import torch
from models import *
from dataset_loader import *
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn.metrics
import optuna

BATCH_SIZE = 32
lr = 0.0005
hidden_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_probs():
    body_loader, eye_loader, face_loader = get_test_loader_eye(25)
    #body_loader, _ = get_test_loader_head()

    head_lstm = GRUModel(input_size = 20, hidden_size = 64, num_layers = 3).to(device)
    head_lstm.load_state_dict(torch.load('./weights/model_head_banging_2.pt', map_location = 'cpu'))
    head_lstm.eval()

    GRU_eyes = ModifiedGRUModel(input_size = 8, hidden_size = 64, num_layers = 4).to(device)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))
    GRU_eyes.eval()

    face = ModifiedGRUModel(input_size = 1434, hidden_size = 64, num_layers = 2).to(device)
    face.load_state_dict(torch.load('./weights/model_face.pt', map_location = 'cpu'))
    face.eval()

    head_output=[]
    eye_output=[]
    face_output = []
    final_labels_1 = []
    final_labels_2 = []
    final_labels_3 = []

    for head_inputs, labels, head_input_lengths in body_loader:
        head_inputs = head_inputs
        labels = labels
        y_head_pred = head_lstm(head_inputs, head_input_lengths)
        for pred in y_head_pred:
            head_output.append(pred.detach().numpy()[1])
        for label in labels:
            final_labels_1.append(label.item())


    for inputs, labels, input_lengths in eye_loader:
        inputs = inputs
        labels = labels
        y_eye_pred = GRU_eyes(inputs, input_lengths)
        for pred in y_eye_pred:
            eye_output.append(pred.detach().numpy()[1])
        for label in labels:
            final_labels_2.append(label.item())

    for inputs, labels, input_lengths in face_loader:
        inputs = inputs
        labels = labels
        y_face_pred = face(inputs, input_lengths)
        for pred in y_face_pred:
            face_output.append(pred.detach().numpy()[1])
        for label in labels:
            final_labels_3.append(label.item())


    return head_output, eye_output, face_output, final_labels_1, final_labels_2, final_labels_3


def late_fusion_probs():
    
    probs_1, probs_2, preds_3, labels_1, labels_2, labels_3  = get_probs()

    final_prob = [(prob_1 + probs_2[i] + preds_3[i])/3 for i, prob_1 in enumerate(probs_1)]
    acc = sklearn.metrics.accuracy_score([round(prob) for prob in final_prob], labels_1)
    df = pd.read_csv('/Users/colinkalicki/Desktop/autism_digital_profiling/ensembling_methods/full_df.csv')
    df['int_fusion'] = final_prob
    df['late_fusion'] = probs_1
    df['head_preds'] = probs_1
    df['eye_results'] = probs_2
    df['face_results'] = preds_3
    df['labels'] = labels_1
    probs_1 = [round(prob) for prob in probs_1]
    probs_2 = [round(prob) for prob in probs_2]
    probs_3 = [round(prob) for prob in preds_3]
    acc_1 = sklearn.metrics.accuracy_score(probs_1, labels_1)
    acc_2 = sklearn.metrics.accuracy_score(probs_2, labels_1)
    acc_3 = sklearn.metrics.accuracy_score(probs_3, labels_1)

    
    df.to_csv('full_df.csv')



def train_late_fusion(epochs, model, batch_size, optimizer):
    head_loader_train, eye_loader_train = get_train_loader(batch_size)
    head_loader_val, eye_loader_val = get_val_loader(25)
    head_loader_test, eye_loader_test = get_test_loader_eye()

    late_probs_model = model
    late_probs_model.train()
    criterion = nn.BCELoss()
    optimizer = optimizer
    total_loss = 0
    total_correct = 0
    total_samples = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        head_progress_bar = tqdm(enumerate(head_loader_train), total=len(head_loader_train), desc="Training")
        eye_loader_train_iter = iter(eye_loader_train)
        for i, (head_inputs, labels, head_input_lengths) in head_progress_bar:
            head_inputs = head_inputs
            labels = labels
            eye_inputs, labels2, eye_input_lengths = next(eye_loader_train_iter)
            inputs = [head_inputs, eye_inputs]
            input_lengths = [head_input_lengths, eye_input_lengths]
            optimizer.zero_grad()

            # Forward pass
            outputs = late_probs_model(inputs, input_lengths)
            loss = criterion(outputs.float(), labels.float())
            total_loss += loss.item() * head_inputs.size(0)

            # Calculate accuracy
            predicted = torch.round(outputs) # Use torch.max to get the index of the max log-probability.
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            head_progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(head_loader_train.dataset)
        epoch_acc = total_correct / total_samples
        print(f'Epoch: [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # Append the loss for this epoch to the list of losses
    train_losses.append(epoch_loss)
    misclassifications = {0: 0, 1: 0} 
    test_correct=0
    test_total = 0
    test_loss = 0
    test_accuracy = 0
    late_probs_model.eval()
    head_progress_bar = tqdm(enumerate(head_loader_test), total=len(head_loader_test), desc="Training")
    eye_loader_train_iter = iter(eye_loader_test)
    for i, (head_inputs, labels, head_input_lengths) in head_progress_bar:
        head_inputs = head_inputs
        labels = labels
        eye_inputs, labels2, eye_input_lengths = next(eye_loader_train_iter)
        inputs = [head_inputs, eye_inputs]
        input_lengths = [head_input_lengths, eye_input_lengths]

        test_outputs = late_probs_model(inputs, input_lengths)
        labels = labels.float()  # Convert labels to long

        test_predicted = torch.round(test_outputs)  # get the index of the max log-probability
        test_correct += (test_predicted == labels).sum().item()
        test_total += labels.size(0)

            # Track misclassifications per class
        for i in range(labels.size(0)):
            if test_predicted[i] != labels[i]:
                misclassifications[int(labels[i].item())] += 1

            #Compute the loss and accumulate it
        loss = criterion(test_outputs, labels)
        test_loss += loss.item() * head_inputs.size(0)

    #Calculate the average test loss over all batches
    test_loss = test_loss / len(head_loader_test.dataset)
    test_accuracy = test_correct / test_total

       
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    for class_label, misclassified_count in misclassifications.items():
        print(f'Misclassifications for class {class_label}: {misclassified_count}')
    
    return test_accuracy

def train_int_fusion(epochs, model, batch_size, optimizer, train_loader, val_loader, test_loader):
    head_loader_train, eye_loader_train, face_loader_train = train_loader
    head_loader_val, eye_loader_val, face_loader_val= val_loader
    head_loader_test, eye_loader_test, face_loader_test = test_loader

    late_probs_model = model
    late_probs_model.train()
    criterion = nn.BCELoss()
    optimizer = optimizer
    total_loss = 0
    total_correct = 0
    total_samples = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_loss = 0
        head_progress_bar = tqdm(enumerate(head_loader_train), total=len(head_loader_train), desc="Training")
        eye_loader_train_iter = iter(eye_loader_train)
        face_loader_train_iter = iter(face_loader_train)

        late_probs_model.train()
        
        for i, (head_inputs, labels, head_input_lengths) in head_progress_bar:
            head_inputs = head_inputs
            labels = labels.to(device)
            eye_inputs, labels2, eye_input_lengths = next(eye_loader_train_iter)
            face_inputs, labels2, face_input_lengths = next(face_loader_train_iter)
            inputs = [head_inputs.to(device), eye_inputs.to(device), face_inputs.to(device)]
            input_lengths = [head_input_lengths.to(device), eye_input_lengths.to(device), face_input_lengths.to(device)]
            optimizer.zero_grad()

            # Forward pass
            outputs = late_probs_model(inputs, input_lengths)
            loss = criterion(outputs.float(), labels.float())
            total_loss += loss.item() * head_inputs.size(0)

            # Calculate accuracy
            predicted = torch.round(outputs) # Use torch.max to get the index of the max log-probability.
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            head_progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(head_loader_train.dataset)
        epoch_acc = total_correct / total_samples
        print(f'Epoch: [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        head_progress_bar = tqdm(enumerate(head_loader_val), total=len(head_loader_val), desc="Training")
        eye_loader_train_iter = iter(eye_loader_val)
        test_correct=0
        test_total = 0
        val_loss = 0
        misclassifications = {0: 0, 1: 0} 
        test_accuracy = 0
        late_probs_model.eval()
        eye_loader_val_iter = iter(eye_loader_val)
        face_loader_val_iter = iter(face_loader_val)
        
        for i, (head_inputs, labels, head_input_lengths) in head_progress_bar:
            head_inputs = head_inputs
            labels = labels
            eye_inputs, labels2, eye_input_lengths = next(eye_loader_val_iter)
            face_inputs, labels2, face_input_lengths = next(face_loader_val_iter)
            inputs = [head_inputs.to(device), eye_inputs.to(device), face_inputs.to(device)]
            input_lengths = [head_input_lengths.to(device), eye_input_lengths.to(device), face_input_lengths.to(device)]

            test_outputs = late_probs_model(inputs, input_lengths)
            labels = labels.float()  # Convert labels to long

            test_predicted = torch.round(test_outputs)  # get the index of the max log-probability
            test_correct += (test_predicted == labels).sum().item()
            test_total += labels.size(0)

            # Track misclassifications per class
            for i in range(labels.size(0)):
                if test_predicted[i] != labels[i]:
                    misclassifications[int(labels[i].item())] += 1

            #Compute the loss and accumulate it
            loss = criterion(test_outputs, labels)
            val_loss += loss.item() * head_inputs.size(0)

        #Calculate the average test loss over all batches
        val_loss = val_loss / len(head_loader_val.dataset)
        test_accuracy = test_correct / test_total
        # Append the loss for this epoch to the list of losses
        train_losses.append(epoch_loss)
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {test_accuracy:.4f}')

    misclassifications = {0: 0, 1: 0} 
    test_correct=0
    test_total = 0
    test_loss = 0
    test_accuracy = 0
    late_probs_model.eval()
    head_progress_bar = tqdm(enumerate(head_loader_test), total=len(head_loader_test), desc="Training")
    eye_loader_train_iter = iter(eye_loader_test)
    eye_loader_test_iter = iter(eye_loader_test)
    face_loader_test_iter = iter(face_loader_test)
        
    for i, (head_inputs, labels, head_input_lengths) in head_progress_bar:
        head_inputs = head_inputs
        labels = labels
        eye_inputs, labels2, eye_input_lengths = next(eye_loader_test_iter)
        face_inputs, labels2, face_input_lengths = next(face_loader_test_iter)
        inputs = [head_inputs.to(device), eye_inputs.to(device), face_inputs.to(device)]
        input_lengths = [head_input_lengths.to(device), eye_input_lengths.to(device), face_input_lengths.to(device)]

        test_outputs = late_probs_model(inputs, input_lengths)
        labels = labels.float()  # Convert labels to long

        test_predicted = torch.round(test_outputs)  # get the index of the max log-probability
        test_correct += (test_predicted == labels).sum().item()
        test_total += labels.size(0)

            # Track misclassifications per class
        for i in range(labels.size(0)):
            if test_predicted[i] != labels[i]:
                misclassifications[int(labels[i].item())] += 1

            #Compute the loss and accumulate it
        loss = criterion(test_outputs, labels)
        test_loss += loss.item() * head_inputs.size(0)

    #Calculate the average test loss over all batches
    test_loss = test_loss / len(head_loader_test.dataset)
    test_accuracy = test_correct / test_total

       
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    for class_label, misclassified_count in misclassifications.items():
        print(f'Misclassifications for class {class_label}: {misclassified_count}')

    return val_loss

#late fusion model based on accuracies of individual prediction
def accuracy_based_ensemble(predictions_list, true_labels):
    """
    Performs accuracy-based ensemble by combining predictions from multiple models.
    
    Args:
        predictions_list (list): List of numpy arrays containing predictions from each model.
        true_labels (numpy array): Ground truth labels.
        
    Returns:
        ensemble_prediction (numpy array): Ensemble prediction.
        ensemble_accuracy (float): Accuracy of the ensemble prediction.
    """
    # Calculate accuracies of each model
    accuracies = [np.mean(predictions == true_labels) for predictions in predictions_list]

    # Calculate weights based on accuracies
    weights = np.array(accuracies) / np.sum(accuracies)

    # Make ensemble prediction
    ensemble_prediction = np.average(predictions_list, axis=0, weights=weights)
    ensemble_prediction_rounded = np.round(ensemble_prediction).astype(int)

    # Calculate accuracy of the ensemble prediction
    ensemble_accuracy = np.mean(ensemble_prediction_rounded == true_labels)

    return ensemble_prediction_rounded, ensemble_accuracy

def objective(trial):
    """Function to be optimized by Optuna.
    Args:
        trial: the current trial
        """
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [15, 20, 25, 31])
    num_epochs = trial.suggest_int('num_epochs', 15, 50)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    hidden_size =  64 
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    

    # Create the model
    
    head_lstm = GRUModel(input_size = 20, hidden_size = 64, num_layers = 3).to(device)
    head_lstm.load_state_dict(torch.load('./weights/best_model_head.pt', map_location = 'cpu'))
    

    GRU_eyes = ModifiedGRUModel(input_size = 8, hidden_size = 64, num_layers = 4).to(device)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))

    model = late_fusion_linear(4, hidden_size, 1, GRU_eyes, head_lstm).to(device)
    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train and evaluate the model, then return the validation loss
    val_loss = train_late_fusion(num_epochs, model, batch_size, optimizer)

    return val_loss

def objective_int(trial, train_loader, val_loader, test_loader):
    """Function to be optimized by Optuna.
    Args:
        trial: the current trial
        """
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [15, 20, 25, 30])
    num_epochs = trial.suggest_int('num_epochs', 20, 60)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    hidden_size =  64
    dropout = trial.suggest_uniform('dropout', 0.1, 0.6)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
   

    # Create the model
    
    head_lstm = GRUModel_last_output(input_size = 20, hidden_size = 64, num_layers = 3).to(device)
    head_lstm.load_state_dict(torch.load('./weights/model_head_banging_2.pt', map_location = 'cpu'))
    

    GRU_eyes = ModifiedGRUModel_hidden_output(input_size = 8, hidden_size = 64, num_layers = 4).to(device)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))

    face_gru = ModifiedGRUModel_hidden_output(input_size = 1434, hidden_size = 64, num_layers = 2).to(device)
    face_gru.load_state_dict(torch.load('./weights/model_face.pt', map_location = 'cpu'))

    model = late_fusion_hidden_layer(hidden_size*3, hidden_size, 1, GRU_eyes, head_lstm, face_gru, dropout).to(device)
    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    val_loss = train_int_fusion(num_epochs, model, batch_size, optimizer, train_loader, val_loader, test_loader)



    return val_loss

def hyperparameter_tuning_late(num_trials=50):
    """Function to tune the hyperparameters of the model using Optuna.
    Args:
        num_trials (int): Number of hyperparameter tuning trials to run.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=num_trials)

    print('Best trial:')
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    head_lstm = GRUModel(input_size = 20, hidden_size = 64, num_layers = 3).to(device)
    head_lstm.load_state_dict(torch.load('./weights/model_head_banging_2.pt', map_location = 'cpu'))
    

    GRU_eyes = ModifiedGRUModel(input_size = 8, hidden_size = 64, num_layers = 4).to(device)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))

    best_model = late_fusion_linear(4, hidden_size, 1, GRU_eyes, head_lstm).to(device)
    # Create the optimizer with the best learning rate and weight decay
    optimizer = optim.Adam(best_model.parameters(), lr=trial.params['learning_rate'], weight_decay=trial.params['weight_decay'])

    # Train the model with the best number of epochs
    val_loss = train_late_fusion(trial.params['num_epochs'], best_model, trial.params['batch_size'], optimizer)

   
    print(val_loss)

    

def hyperparameter_tuning_int(num_trials=50):
    """Function to tune the hyperparameters of the model using Optuna.
    Args:
        num_trials (int): Number of hyperparameter tuning trials to run.
    """
    head, face, eye = get_train_loader(25)
    train_loader = [head, face, eye]
    head, face, eye = get_val_loader(25)
    val_loader = [head, face, eye]
    head, face, eye = get_test_loader_eye(25)
    test_loader = [head, face, eye]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_int(trial, train_loader, val_loader, test_loader), n_trials=num_trials)

    print('Best trial:')
    trial = study.best_trial
    print(" Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    
    head_lstm = GRUModel_last_output(input_size = 20, hidden_size = 64, num_layers = 3).to(device)
    head_lstm.load_state_dict(torch.load('./weights/model_head_banging_2.pt', map_location = 'cpu'))
    

    GRU_eyes = ModifiedGRUModel_hidden_output(input_size = 8, hidden_size = 64, num_layers = 4).to(device)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))


    face_gru = ModifiedGRUModel_hidden_output(input_size = 1434, hidden_size = 64, num_layers = 2).to(device)
    face_gru.load_state_dict(torch.load('./weights/model_face.pt', map_location = 'cpu'))

    best_model = late_fusion_hidden_layer(hidden_size*3, hidden_size, 1, GRU_eyes, head_lstm, face_gru, trial.params['dropout']).to(device)


    # Create the optimizer with the best learning rate and weight decay
    optimizer = optim.Adam(best_model.parameters(), lr=trial.params['learning_rate'], weight_decay=trial.params['weight_decay'])

    # Train the model with the best number of epochs
    val_loss = train_int_fusion(trial.params['num_epochs'], best_model, trial.params['batch_size'], optimizer)

    print(val_loss)

if __name__ == '__main__':
    torch.manual_seed(0)
    head_lstm = GRUModel(input_size = 20, hidden_size = 64, num_layers = 3)
    head_lstm.load_state_dict(torch.load('./weights/best_model_head.pt', map_location = 'cpu'))

    GRU_eyes = ModifiedGRUModel(input_size = 8, hidden_size = 64, num_layers = 4)
    GRU_eyes.load_state_dict(torch.load('./weights/model_eye_gaze_v3.pt', map_location = 'cpu'))
    late_fusion_probs()
    #train_late_fusion(30, GRU_eyes, head_lstm)
    hyperparameter_tuning_int(50)
    hyperparameter_tuning_late(50)
