import tensorflow as tf
import numpy as np

# Define the user's transaction history
transaction_history = np.array([
    [100.0, 'Food'],
    [50.0, 'Shopping'],
    [80.0, 'Food'],
    [30.0, 'Entertainment'],
    [120.0, 'Travel'],
    ...
])

# Create a vocabulary of unique transaction categories
categories = np.unique(transaction_history[:, 1])
num_categories = len(categories)

# Map each category to a unique integer ID
category_to_id = {category: i for i, category in enumerate(categories)}

# Convert transaction history into numerical sequences
sequence = [category_to_id[transaction[1]] for transaction in transaction_history]

# Define the input and target data for training
input_data = sequence[:-1]
target_data = sequence[1:]

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(num_categories, 16, input_length=1),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(num_categories, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the neural network
model.fit(np.expand_dims(input_data, axis=-1), target_data, epochs=100)

# Generate a new sequence of transactions based on user behavior
start_category = 'Food'
num_predictions = 10

generated_sequence = [category_to_id[start_category]]
for _ in range(num_predictions):
    input_sequence = np.array(generated_sequence[-1]).reshape(1, 1)
    predicted_category = model.predict(input_sequence).argmax()
    generated_sequence.append(predicted_category)

# Convert the generated sequence back to transaction categories
generated_transactions = [categories[category] for category in generated_sequence]

print("Generated Transactions:")
print(generated_transactions)
