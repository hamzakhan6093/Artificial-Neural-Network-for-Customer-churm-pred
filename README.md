# Artificial-Neural-Network-for-Customer-churm-pred
(ANN for Customer Churn Prediction)
1. Dataset Overview
File: Churn_Modelling.csv
Description: The dataset contains customer information, including demographics, account details, and whether they exited the bank (Exited column).
Purpose: To predict customer churn using an Artificial Neural Network (ANN).
2. Preprocessing the Data
Label Encoding:

Columns like Gender and Geography are categorical and need to be converted into numerical formats.
Use LabelEncoder for binary categories (Gender) and OneHotEncoder for multi-class categories (Geography).
Feature Scaling:

Normalize or standardize numerical columns (e.g., CreditScore, Balance, EstimatedSalary) to bring them into a uniform range for faster convergence during training.
Use StandardScaler or similar scaling methods.
Splitting the Data:

Divide the data into training and testing sets. For instance, 80% for training and 20% for testing.
Use train_test_split from sklearn.
3. Building the ANN
Input Layer:

Represents the number of features in the data (after preprocessing). For example, if there are 12 features after encoding and scaling, the input layer will have 12 nodes.
Hidden Layers:

Add one or more fully connected (dense) layers with activation functions such as ReLU.
The number of neurons in each hidden layer depends on experimentation but can start with values like 6, 12, or a factor of the input features.
Output Layer:

For binary classification, the output layer has 1 neuron with a sigmoid activation function, which outputs a probability value.
4. Compiling the ANN
Loss Function:
Use binary_crossentropy for binary classification.
Optimizer:
Use Adam optimizer for efficient gradient descent.
Metrics:
Include metrics like accuracy to evaluate performance during training.
5. Training the ANN
Fitting the Model:
Train the ANN using the fit method on the training set.
Specify the number of epochs (e.g., 50) and batch size (e.g., 32).
Validation:
Monitor performance on the validation or testing set to prevent overfitting.
6. Evaluating the ANN
Test Set Evaluation:
Predict on the test set using the predict method.
Convert probabilities into binary outputs using a threshold (e.g., 0.5).
Performance Metrics:
Use metrics such as confusion_matrix, precision, recall, and F1-score to evaluate performance.
