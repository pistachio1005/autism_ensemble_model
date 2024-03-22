import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler



def load_test_data():
    """This function loads the data from the csv files."""
    #We load the train, validation and test data
    df = pd.read_csv('./splits/test_eye_head_face.csv')
    #df_eye = pd.read_csv('./splits/train_eye.csv')

    #We convert the 'eye_gazing_features' column from string to list
    #df_test['face_features'] = df_face['face_features'].apply(eval)
    df['head_features'] = df['head_features'].apply(eval)
    df['eye_gazing_features'] = df['eye_gazing_features'].apply(eval)
    df['face_feature'] = df['face_feature'].apply(eval)

    #X_face = np.array(df_test['eye_gazing_features'].to_list(), dtype=object)
    #y_face = np.array(df_test['ASD'])

    X_head = np.array(df['head_features'].to_list(), dtype=object)
    y_head = np.array(df['ASD'])

    X_eye = np.array(df['eye_gazing_features'].to_list(), dtype=object)
    y_eye = np.array(df['ASD'])

    X_face = np.array(df['face_feature'].to_list(), dtype=object)
    y_face= np.array(df['ASD'])

    return X_head, y_head, X_eye, y_eye, X_face, y_face


def load_val_data():
    """This function loads the data from the csv files."""
    df = pd.read_csv('./splits/val_eye_head_face.csv')
    #df_eye = pd.read_csv('./splits/train_eye.csv')

    #We convert the 'eye_gazing_features' column from string to list
    #df_test['face_features'] = df_face['face_features'].apply(eval)
    df['head_features'] = df['head_features'].apply(eval)
    df['eye_gazing_features'] = df['eye_gazing_features'].apply(eval)
    df['face_feature'] = df['face_feature'].apply(eval)

    #X_face = np.array(df_test['eye_gazing_features'].to_list(), dtype=object)
    #y_face = np.array(df_test['ASD'])

    X_head = np.array(df['head_features'].to_list(), dtype=object)
    y_head = np.array(df['ASD'])

    X_eye = np.array(df['eye_gazing_features'].to_list(), dtype=object)
    y_eye = np.array(df['ASD'])

    X_face = np.array(df['face_feature'].to_list(), dtype=object)
    y_face= np.array(df['ASD'])

    return X_head, y_head, X_eye, y_eye, X_face, y_face


def load_train_data():
    """This function loads the data from the csv files."""
    #We load the train, validation and test data
    df = pd.read_csv('./splits/train_eye_head_face.csv')
    #df_eye = pd.read_csv('./splits/train_eye.csv')

    #We convert the 'eye_gazing_features' column from string to list
    #df_test['face_features'] = df_face['face_features'].apply(eval)
    df['head_features'] = df['head_features'].apply(eval)
    df['eye_gazing_features'] = df['eye_gazing_features'].apply(eval)
    df['face_feature'] = df['face_feature'].apply(eval)

    #X_face = np.array(df_test['eye_gazing_features'].to_list(), dtype=object)
    #y_face = np.array(df_test['ASD'])

    X_head = np.array(df['head_features'].to_list(), dtype=object)
    y_head = np.array(df['ASD'])

    X_eye = np.array(df['eye_gazing_features'].to_list(), dtype=object)
    y_eye = np.array(df['ASD'])

    X_face = np.array(df['face_feature'].to_list(), dtype=object)
    y_face= np.array(df['ASD'])
    
    return X_head, y_head, X_eye, y_eye, X_face, y_face

class dataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.from_numpy(np.array(x)).float() for x in X]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def collate_fn(batch):
    # Unzip the batch
    X, y = zip(*batch)
    # Get sequence lengths
    X_lengths = torch.tensor([len(x) for x in X])
    # Pad the sequences
    X_padded = pad_sequence([torch.FloatTensor(x) for x in X], batch_first=True, padding_value=0)
    # Convert labels to tensor
    y = torch.tensor(y).long()
    
    return X_padded, y, X_lengths 

def get_test_loader_eye(batch_size=25, normalize=True):
    """This function returns the dataloaders for the train, validation and test data."""
    # Load the data
    X_body, y_body, X_eye, y_eye, X_face, y_face = load_test_data()
    #face_dataset = dataset(X_face, y_face)
    body_dataset = dataset(X_body, y_body)
    eye_dataset = dataset(X_eye, y_eye)
    face_dataset = dataset(X_face, y_face)

    # Create dataloaders
    #face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    body_loader = DataLoader(body_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    eye_loader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return body_loader, eye_loader, face_loader

def get_test_loader_head(batch_size=20, normalize=True):
    """This function returns the dataloaders for the train, validation and test data."""
    # Load the data
    X_body, y_body, X_eye, y_eye = load_test_data()
    #face_dataset = dataset(X_face, y_face)
    body_dataset = dataset(X_body, y_body)
    eye_dataset = dataset(X_eye, y_eye)

    # Create dataloaders
    #face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    body_loader = DataLoader(body_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    eye_loader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return body_loader, eye_loader

def get_val_loader(batch_size=25, normalize=True):
    """This function returns the dataloaders for the train, validation and test data."""
    # Load the data
    X_body, y_body, X_eye, y_eye, X_face, y_face = load_val_data()
    #face_dataset = dataset(X_face, y_face)
    body_dataset = dataset(X_body, y_body)
    eye_dataset = dataset(X_eye, y_eye)
    face_dataset = dataset(X_face, y_face)

    # Create dataloaders
    #face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    body_loader = DataLoader(body_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    eye_loader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return body_loader, eye_loader, face_loader


def get_train_loader(batch_size=25, normalize=True):
    """This function returns the dataloaders for the train, validation and test data."""
    # Load the data
    X_body, y_body, X_eye, y_eye, X_face, y_face = load_train_data()
    #face_dataset = dataset(X_face, y_face)
    body_dataset = dataset(X_body, y_body)
    eye_dataset = dataset(X_eye, y_eye)
    face_dataset = dataset(X_face, y_face)

    # Create dataloaders
    #face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    body_loader = DataLoader(body_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    eye_loader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    face_loader = DataLoader(face_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return body_loader, eye_loader, face_loader

def class_weights(train_loader, num_classes=2):
    """This function computes the class weights for the training data to feed into the loss function."""
    # Assume train_loader is your DataLoader
    class_counts = torch.zeros(num_classes)
    # Count the occurrences of each class in the DataLoader
    for _, labels, _ in train_loader:
        class_counts += torch.bincount(labels, minlength=num_classes)
    # Compute the class frequencies and class weights
    class_frequencies = class_counts / class_counts.sum()
    class_weights = 1.0 / (num_classes * class_frequencies)
    # Normalize the class weights
    class_weights /= class_weights.sum()
    return class_weights

def normalize_with_padding(data, padding_value=0):
    """
    Normalizes data considering padding values.
    Args:
        data (torch.Tensor): tensor with the shape (batch_size, seq_length, num_features)
        padding_value (float): the padding value in the data
    Returns:
        torch.Tensor: normalized data
    """
    # Create a mask for non-padded data
    mask = data != padding_value
    # StandardScaler expects a 2D array, reshape your data
    reshaped_data = data[mask].reshape(-1, 1)
    # Fit and transform the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(reshaped_data)
    # Reshape the normalized_data to match the shape of data[mask]
    normalized_data = normalized_data.reshape(-1)
    # Replace the non-padded values with the normalized values
    data[mask] = torch.Tensor(normalized_data).to(data.device)
    return data


 
class BodyposDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target


