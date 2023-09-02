import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import dump

# Define mechanical properties that you want to predict
mechanical_properties = ['hardness', 'yield', 'tensile', 'elongation', ]


def get_target_number(mechanical):
    """Maps a mechanical property to its corresponding column index."""
    target_dict = {
        'hardness': 2,
        'yield': 3,
        'tensile': 4,
        'elongation': 5,
    }
    return target_dict.get(mechanical)


def create_training_set(test_size, random_state, mechanical):
    """Create a training set for the prediction based on alloy composition and processing method."""
    target_number = get_target_number(mechanical)

    # Read the parsed mechanical dataset and transform it into NumPy arrays
    mech_data = pd.read_excel('../dataset/MPEA_parsed_dataset.xlsx')
    mech_array = mech_data.to_numpy()

    # Identify the features and target for prediction
    composition = mech_array[:, 10:53].astype(float)
    target = mech_array[:, target_number].astype(float)

    # Identify and remove the entries containing NaN target values
    valid_index = ~np.isnan(target)
    x = composition[valid_index]
    y = target[valid_index]

    # Split the dataset into train and test sets
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_and_evaluate_regressor(mechanical_property, split, rand_dataset):
    """Train and evaluate a Random Forest Regressor for a given mechanical property."""
    X_train, X_test, y_train, y_test = create_training_set(split, rand_dataset, mechanical_property)
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=0, oob_score=True)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print(f'R2 score of RF prediction on {mechanical_property} is: {r2}')

    # Save the trained model
    dump(rf_regressor, f'../saved_models/{mechanical_property}_regressor.joblib')
    return r2


if __name__ == "__main__":
    split = 0.1
    rand_dataset = 49
    for i in range(len(mechanical_properties)):
        mechanical_property = mechanical_properties[i]
        train_and_evaluate_regressor(mechanical_property, split, rand_dataset)



