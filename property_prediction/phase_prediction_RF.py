import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

property_list = ['FCC', 'BCC', 'HCP', 'IM']


def get_target_number(prop):
    """Maps a property to its corresponding column index."""
    target_dict = {
        'FCC': 10,
        'BCC': 11,
        'HCP': 12,
        'IM': 13
    }
    return target_dict.get(prop, 2)  # default to 'hardness'


def create_training_set(test_size, random_state, prop):
    """Create a training set for the prediction of phase based on alloy composition and processing method."""
    target_number = get_target_number(prop)

    # Read and transform the parsed mechanical dataset into NumPy arrays
    mech_data = pd.read_excel('../dataset/MPEA_parsed_dataset.xlsx')
    mech_array = mech_data.to_numpy()

    # Identify the features and target for prediction
    composition = mech_array[:, 14:53].astype(float)
    target = mech_array[:, target_number].astype(float)

    # Identify and remove the entries containing NaN target values
    valid_index = ~np.isnan(target)
    x = composition[valid_index]
    y = target[valid_index]

    # Split the dataset into train and test sets
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_and_evaluate_classifier(s, split, rand_dataset, rand_classify):
    """Train and evaluate a Random Forest Classifier for a given property."""
    X_train, X_test, y_train, y_test = create_training_set(split, rand_dataset, property_list[s])
    rf_classifier = RandomForestClassifier(max_depth=50, n_estimators=100, random_state=rand_classify, oob_score=True)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    dump(rf_classifier, f'../saved_models/{property_list[s]}_classifier.joblib')
    print(f'Accuracy of RF prediction on {property_list[s]} is: {accuracy}')


if __name__ == "__main__":
    split = 0.1
    rand_classify = 4
    rand_dataset = 14

    for s in range(4):  # Corresponds to 'FCC', 'BCC', 'HCP', 'IM'
        train_and_evaluate_classifier(s, split, rand_dataset, rand_classify)
