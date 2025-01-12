# MNIST Neural Network Classifier with Flask API

This project implements a simple neural network classifier for the MNIST dataset using Python. The model is trained using the `MLPClassifier` from `scikit-learn`, and a Flask API is provided to make predictions on handwritten digit images.

---

## Features

- **Dataset**: MNIST handwritten digits dataset.
- **Model**: Neural network with one hidden layer, implemented using `MLPClassifier`.
- **Data Preprocessing**:
  - Normalizes pixel values to improve model performance.
  - Splits the dataset into training and testing sets.
- **Evaluation**:
  - Includes cross-validation to assess model generalization.
  - Monitors training loss during training.
- **Deployment**:
  - Flask API to accept digit images as input and return predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shahad-irl/mnist-classifier.git
   cd mnist-classifier
   ```

2. Install the required Python packages:
   ```bash
   pip install numpy scikit-learn flask matplotlib
   ```

---
## Description

1. **Dataset Loading**:
   - The code attempts to load the MNIST dataset using three different sources:
     - `fetch_openml` from scikit-learn.
     - `load_digits` from scikit-learn as a fallback.
     - Keras MNIST dataset if the first two fail.
   - The dataset is loaded with error handling to ensure the program continues by falling back to alternative datasets when needed.

2. **Data Preprocessing**:
   - The pixel values are normalized using `StandardScaler` from scikit-learn.
   - The dataset is split into training and testing sets using `train_test_split`.

3. **Model Training**:
   - A neural network model (`MLPClassifier`) is defined with one hidden layer of 128 neurons, ReLU activation, and the Adam optimizer.
   - The model is trained using 5-fold cross-validation and the training data.

4. **Model Evaluation**:
   - The model is evaluated on the test set to calculate accuracy.
   - The training loss curve is plotted to visualize the model's performance during training.

5. **Flask API**:
   - A simple Flask API is set up with a `/predict` endpoint that accepts a POST request with a JSON object containing the input data (flattened 28x28 image).
   - The API returns the predicted digit based on the trained model.

---
## Usage

### Training the Model

1. Run the script to train the model:
   ```bash
   python mnist-classifier.py
   ```

2. During training, the script will:
   - Perform 5-fold cross-validation.
   - Train the model on the MNIST dataset.
   - Output training loss and accuracy.

### Running the Flask API

1. Start the Flask server:
   ```bash
   python mnist-classifier.py
   ```

2. Use an API testing tool (e.g., Postman) or a Python script to send POST requests to the `/predict` endpoint.
   - Example input:
     ```json
     {
       "data": [0, 0, 0, ..., 255]  // Flattened 28x28 array of pixel values
     }
     ```
   - Example response:
     ```json
     {
       "prediction": 5
     }
     ```

---

## File Structure

```
mnist-classifier/
├── mnist_classifier.py  # Main script
├── README.md            # Project documentation
```

---
## Results

- **Cross-Validation Accuracy**: Displayed during training.

- **Test Accuracy**: Achieved after training, evaluated on a separate test set.

- **Training Loss Curve**: Plots the loss during training to monitor performance.
  - Expected output:
    ```
    Cross-validation scores: [0.84375    0.84722222 0.80836237 0.84320557 0.82926829]
    Average cross-validation score: 0.834361691831204
    Test Accuracy: 89.17%
    ```
    ![Training Loss Curve](https://github.com/user-attachments/assets/ece0a553-3ac6-4d20-8b80-f00bae4e63b2)


---

## Limitations

- The model is designed for educational purposes and might not handle real-world images without preprocessing.
- Flask API expects input as a flattened 28x28 array of pixel values, normalized between 0 and 1.

---


## Future Improvements

- Enhance the neural network by adding more layers or using a deep learning framework (e.g., TensorFlow or PyTorch).
- Integrate image preprocessing for real-world handwritten digit recognition.
- Dockerize the Flask API for easier deployment.

---

## Author

**Shahad-irl**  
A computer engineering graduate passionate about artificial intelligence and machine learning.
