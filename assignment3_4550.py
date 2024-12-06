"""
Assignment 3 - COIS 4550
Liam Atwood
0695281

Implements the Naïve Bayes classification algorithm for datasets with categorical features.
The program:
1) Loads a CSV file with categorical data.
2) Splits the dataset into training (80%) and test (20%) sets.
3) Classifies test instances using the NB algorithm.
4) Calculates and prints the accuracy.
"""

import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------- Naïve Bayes Classifier ---------------------------------------

class NaiveBayesClassifier:
    """Naïve Bayes classifier for categorical data."""

    def __init__(self):
        self.priors = defaultdict(float)
        self.likelihoods = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes = []

    def train(self, X, y):
        """Train the Naïve Bayes model."""
        self.classes = y.unique()
        total_instances = len(y)

        # Calculate priors P(C)
        for c in self.classes:
            self.priors[c] = len(y[y == c]) / total_instances

        # Calculate likelihoods P(F_i | C)
        for feature in X.columns:
            for c in self.classes:
                feature_counts = X[y == c][feature].value_counts()
                total_class_instances = len(X[y == c])
                for value, count in feature_counts.items():
                    self.likelihoods[feature][value][c] = count / total_class_instances

    def predict(self, X):
        """Predict the class for each instance in X."""
        predictions = []
        for _, instance in X.iterrows():
            class_probabilities = {}
            for c in self.classes:
                # Start with the prior
                class_probabilities[c] = self.priors[c]
                # Multiply by likelihoods
                for feature, value in instance.items():
                    class_probabilities[c] *= self.likelihoods[feature][value].get(c, 1e-6)

            # Choose the class with the highest probability
            predictions.append(max(class_probabilities, key=class_probabilities.get))
        return predictions


# ------------------------------------ Main Program ----------------------------------------

def main():
    """Main program implementation."""

    # Load data
    file_path = input("Enter the path to the CSV file: ")
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")

    # Split into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train)
    print("Training complete.")

    # Predict on test set
    predictions = nb_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Set Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
