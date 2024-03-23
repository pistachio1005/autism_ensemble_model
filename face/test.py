import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Function to process the face_features column and convert it to a numpy array
def process_features(features_series):
    # Convert string representation of lists to actual lists
    processed_features = features_series.apply(lambda x: np.array(eval(x)))
    # Stack to create a single numpy array
    return np.stack(processed_features.values)

# Load your data
train_df = pd.read_csv('/Users/yingsun/Documents/GitHub/ensembling_methods/facial_landmarks/splits/train_padded.csv')
val_df = pd.read_csv('/Users/yingsun/Documents/GitHub/ensembling_methods/facial_landmarks/splits/val_padded.csv')
test_df = pd.read_csv('/Users/yingsun/Documents/GitHub/ensembling_methods/facial_landmarks/splits/test_padded.csv')

# Process face_features and labels
X_train = process_features(train_df['face_features'])
X_val = process_features(val_df['face_features'])
X_test = process_features(test_df['face_features'])

y_train = train_df['ASD'].values
y_val = val_df['ASD'].values
y_test = test_df['ASD'].values

# Assuming face_features are already padded and reshaped as needed for the model input
# If necessary, adjust the reshaping based on your specific data structure
# Example: X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1, 1)

# Create the model
model = Sequential([
    TimeDistributed(Conv1D(32, kernel_size=3, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=16, epochs=8, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Generate predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")  # Convert probabilities to binary class labels

# Summarize predictions with a confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=['Non-ASD', 'ASD']))

