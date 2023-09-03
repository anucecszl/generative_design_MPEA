import numpy as np
import torch
import WGAN_GP
import pandas
from joblib import load

GEN_PATH = '../saved_models/generator_net_MPEA.pt'
DATA_PATH = '../dataset/MPEA_parsed_dataset.xlsx'
RESULT_PATH = '../generated_datasets/generated_MPEA_WGAN_GP.xlsx'

# Read element, process, and feature names from the dataset
element_names = pandas.read_excel(DATA_PATH).columns.values[14:46]
process_names = pandas.read_excel(DATA_PATH).columns.values[46:53]
feature_names = pandas.read_excel(DATA_PATH).columns.values[14:53]

# Load the dataset into a DataFrame
dataset_df = pandas.read_excel(DATA_PATH)

# Convert DataFrame to NumPy array for manipulation
data_np = dataset_df.to_numpy()

# Extract alloy features (composition data) from the dataset
comp_data = data_np[:, 14:46].astype(float)

# Find the minimum and maximum values for normalization
comp_min = comp_data.min(axis=0)
comp_max = comp_data.max(axis=0)

# Set the number of samples to be generated
num_samples = 10000

# Load the pretrained WGAN generator model
generator = WGAN_GP.Generator()
generator.load_state_dict(torch.load(GEN_PATH))
generator.eval()

# Generate synthetic alloy compositions using random noise
g_noise = torch.tensor(np.random.randn(num_samples, 10)).float()
fake_alloys = generator(g_noise).detach().numpy()

# De-normalize and standardize the generated compositions
fake_alloys[:, :32] = fake_alloys[:, :32] * comp_max + comp_min
fake_alloys[:, :32] = fake_alloys[:, :32] / np.sum(fake_alloys[:, :32], axis=1).reshape((-1, 1))

alloy_names = []
element_lists = []
ratio_lists = []

for i in range(fake_alloys.shape[0]):
    composition = fake_alloys[i, :32]
    comp_string = ''
    element_list = []
    ratio_list = []
    for j in range(len(composition)):
        # Only keep elements with molar ratio > 0.005
        if composition[j] > 0.005:
            comp_string += element_names[j]
            comp_string += str(round(composition[j], 2))
            element_list.append(element_names[j])
            ratio_list.append(composition[j])
        else:
            composition[j] = 0
    alloy_names.append(comp_string)
    element_lists.append(element_list)
    ratio_lists.append(ratio_list)

# One-hot encode the processing methods for the generated alloys
process_indices = np.argmax(fake_alloys[:, 32:], axis=1)
process_one_hot = np.zeros_like(fake_alloys[:, 32:])

for i in range(len(process_one_hot)):
    for j in range(len(process_one_hot[i])):
        if process_indices[i] == j:
            process_one_hot[i][j] = 1
fake_alloys[:, 32:] = process_one_hot

# Identify the feature and predicted properties of the generated alloys
composition = fake_alloys[:, :32]
processing = fake_alloys[:, 32:]


# Predict phases and mechanical properties for generated alloys
phase_array = np.zeros((num_samples, 4))
classifier_FCC = load('../saved_models/FCC_classifier.joblib')
classifier_BCC = load('../saved_models/BCC_classifier.joblib')
classifier_HCP = load('../saved_models/HCP_classifier.joblib')
classifier_IM = load('../saved_models/IM_classifier.joblib')

feature_for_phase = np.concatenate((composition, processing), axis=1)
phase_array[:, 0] = classifier_FCC.predict(feature_for_phase)
phase_array[:, 1] = classifier_BCC.predict(feature_for_phase)
phase_array[:, 2] = classifier_HCP.predict(feature_for_phase)
phase_array[:, 3] = classifier_IM.predict(feature_for_phase)

print('Finished phase prediction.')

# Identify the predicted phases as strings
phase_name_list = []
phase_name_strings = ['FCC', 'BCC', 'HCP', 'IM']
for i in range(num_samples):
    phase_name = ''
    for j in range(4):
        if phase_array[i][j] > 0:
            phase_name += phase_name_strings[j]
            phase_name += '+'
    phase_name_list.append(phase_name[:-1])

# predict mechanical properties
property_array = np.zeros((num_samples, 4))
regressor_hardness = load('../saved_models/hardness_regressor.joblib')
regressor_yield = load('../saved_models/yield_regressor.joblib')
regressor_tensile = load('../saved_models/tensile_regressor.joblib')
regressor_elongation = load('../saved_models/elongation_regressor.joblib')

feature_for_property = np.concatenate((composition, processing), axis=1)
property_array[:, 0] = regressor_hardness.predict(feature_for_property)
property_array[:, 1] = regressor_yield.predict(feature_for_property)
property_array[:, 2] = regressor_tensile.predict(feature_for_property)
property_array[:, 3] = regressor_elongation.predict(feature_for_property)

correct = (property_array[:, 2] > property_array[:, 1])

print('Finished mechanical property prediction.')

name_list = [x for x, y in zip(alloy_names, correct) if y]
proc_list = [x for x, y in zip(list(process_indices + 1), correct) if y]
phas_list = [x for x, y in zip(phase_name_list, correct) if y]

# Create DataFrames for final output and save it as an Excel file
formula_df = pandas.DataFrame(zip(name_list, proc_list, phas_list),
                              columns=['Composition', 'Processing method', 'Predicted phase'])
property_df = pandas.DataFrame(property_array[correct, :], columns=['Hardness', 'Yield',
                                                                    'Tensile', 'Elongation'])
feature_df = pandas.DataFrame(feature_for_property[correct, :], columns=list(feature_names))

output_df = pandas.concat([formula_df, property_df, feature_df], axis=1)
output_df.to_excel(RESULT_PATH, index=False)
