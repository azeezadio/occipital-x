import tensorflow as tf
import numpy as np

# Define the training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, input_shape=(2,), activation='sigmoid'),  # Input layer with 2 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the neural network
model.fit(x_train, y_train, epochs=1000)

# Test the trained model
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = model.predict(x_test)

print("Predictions:")
for i in range(len(x_test)):
    print(f"Input: {x_test[i]}, Predicted Output: {y_pred[i]}")
