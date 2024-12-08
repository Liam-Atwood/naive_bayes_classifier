"""
Assignment 3 - COIS 4550
Liam Atwood
0695281

Implements the Naïve Bayes classification algorithm for datasets with categorical features.
The program:
1) Loads a CSV file with categorical data.
2) Splits the dataset into training (80%) and test (20%) sets.
3) Classifies test group using the NB algorithm.
4) Calculates the accuracy.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------- Naïve Bayes Classifier ---------------------------------------

class NaiveBayesClassifier:
    """Naïve Bayes classifier for categorical data."""

    def __init__(self):
        self.priors = {}  # Class probabilities (P(C))
        self.likelihoods = {}  # Feature likelihoods (P(F|C))
        self.classes = []  # Unique class labels

    def train(self, x, y):
        """Train the Naïve Bayes model."""
        self.classes = y.unique()
        total_instances = len(y)

        # Calculate P(C)
        self.priors = {c: len(y[y == c]) / total_instances for c in self.classes}

        # Calculate likelihoods P(F|C)
        self.likelihoods = {
            feature: {
                value: {
                    c: (x[y == c][feature] == value).sum() / len(x[y == c])
                    for c in self.classes
                }
                for value in x[feature].unique()
            }
            for feature in x.columns
        }

    def predict(self, x):
        """Predict the class for each instance in X."""
        predictions = []
        for _, instance in x.iterrows():
            class_probabilities = {}
            for c in self.classes:
                # Previous probability
                prob = self.priors[c]
                # Multiply by likelihoods
                for feature, value in instance.items():
                    prob *= self.likelihoods.get(feature, {}).get(value, {}).get(c, 1e-6)
                class_probabilities[c] = prob

            # Predict the highest probability
            predictions.append(max(class_probabilities, key=class_probabilities.get))
        return predictions


# ------------------------------------ Main Program ---------------------------------------------------

def main():
    """Main program implementation."""

    # Load data
    file_path = "heartCategorical.csv"
    data = pd.read_csv(file_path)

    # Split into categorical features and class variable
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(x_train, y_train)
    print("Training complete.")

    # Predict using test set
    predictions = nb_classifier.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Set Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
