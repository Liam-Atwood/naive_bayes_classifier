import pandas as pd


def preprocess_dataset(input_file, output_file):
    """
    Preprocess the dataset:
    - Bin continuous variables into categories.
    - Save the transformed dataset into a new CSV file.

    Args:
    input_file: Path to the input CSV file.
    output_file: Path to save the transformed CSV file.
    """
    # Load the dataset
    data = pd.read_csv(input_file)

    # Bin continuous variables
    data['Age_Group'] = pd.cut(
        data['age'], bins=[0, 30, 40, 50, 60, 70, 80, 100],
        labels=['0-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-100']
    )

    data['BP_Group'] = pd.cut(
        data['trtbps'], bins=[90, 120, 140, 160, 200],
        labels=['Low', 'Normal', 'Elevated', 'High']
    )

    data['Chol_Group'] = pd.cut(
        data['chol'], bins=[100, 200, 240, 300, 400],
        labels=['Ideal', 'Borderline High', 'High', 'Very High']
    )

    data['Heart_Rate_Group'] = pd.cut(
        data['thalachh'], bins=[70, 100, 130, 160, 200],
        labels=['Low', 'Below Average', 'Average', 'Above Average']
    )

    # Drop original continuous columns (optional)
    data.drop(columns=['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'], inplace=True)

    # Reorder columns for clarity
    categorical_columns = ['Age_Group', 'BP_Group', 'Chol_Group', 'Heart_Rate_Group']
    data = data[categorical_columns + [col for col in data.columns if col not in categorical_columns]]

    # Save the transformed dataset
    data.to_csv(output_file, index=False)
    print(f"Transformed dataset saved to {output_file}")


# Example usage
input_csv = "heart.csv"  # Replace with the path to your input CSV file
output_csv = "heartCategorical.csv"  # Replace with the path for the output CSV file
preprocess_dataset(input_csv, output_csv)
