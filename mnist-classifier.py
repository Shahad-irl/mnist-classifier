import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt

# Try loading MNIST dataset with fetch_openml
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, data_home='./', timeout=120)  # Timeout set to 120 seconds
    X = mnist['data'].values
    y = mnist['target'].values.astype(int)
    print("Dataset loaded successfully using fetch_openml!")
except Exception as e:
    print(f"Failed to fetch dataset using fetch_openml: {e}")
    print("Attempting alternative datasets...")
    try:
        # Alternative 1: Use load_digits from sklearn
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data
        y = digits.target
        print("Dataset loaded successfully using load_digits!")
    except Exception as e2:
        print(f"Failed to load dataset using load_digits: {e2}")
        print("Attempting to use Keras MNIST dataset...")
        try:
            # Alternative 2: Use Keras MNIST dataset
            from tensorflow.keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X = X_train.reshape(-1, 28 * 28)
            y = y_train
            print("Dataset loaded successfully using Keras MNIST dataset!")
        except Exception as e3:
            print(f"Failed to load dataset using Keras MNIST dataset: {e3}")
            print("Please check your setup or load the dataset manually.")

# Normalize pixel values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=5, activation='relu', solver='adam')

# Train the model with cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation

# Print cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)}")

# Train the model on the full training data
model.fit(X_train, y_train)

# Evaluate the model on test data
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training loss curve
try:
    plt.plot(model.loss_curve_)
    plt.title('Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
except AttributeError:
    print("Loss curve is not available. Ensure the model is trained.")

# Flask API for deployment
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'data' not in request.json:
        return jsonify({'error': 'No input data provided'}), 400

    data = request.json['data']  # Input should be a 28x28 flattened array
    if len(data) != 784:
        return jsonify({'error': 'Input data must contain 784 values'}), 400
    
    data = np.array(data).reshape(1, -1)  # Normalize input if needed

    # Predict using the model
    prediction = model.predict(data)[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
