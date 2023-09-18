import pandas

# Load the dataset from the Excel file.
dataset = pandas.read_excel('MPEA_mechanical_database.xlsx')

# Extract all column names.
feature = dataset.columns.values

# Extract data for each feature.
alloy_list = list(dataset[feature[1]])
phase_list = list(dataset[feature[2]])
hardness_list = list(dataset[feature[3]])
yield_list = list(dataset[feature[4]])
tensile_list = list(dataset[feature[5]])
elongation_list = list(dataset[feature[6]])
compressive_list = list(dataset[feature[7]])
plasticity_list = list(dataset[feature[8]])
process_list = list(dataset[feature[9]])
source_list = list(dataset[feature[10]])

# Clean up unwanted spaces from the process and source lists.
comment_lists = [process_list, source_list]
for comment_list in comment_lists:
    for i in range(len(comment_list)):
        if isinstance(comment_list[i], str):
            comment_list[i] = comment_list[i].strip()

# Remove any spaces within alloy names.
for i in range(len(alloy_list)):
    alloy_list[i] = alloy_list[i].replace(" ", "")

# Clean up the phase list format.
cleaned_phase_list = []
for i in range(len(phase_list)):
    phase_string = ""
    if isinstance(phase_list[i], str):
        phases = []
        split_list = phase_list[i].split('+')
        for j in split_list:
            # Handle specific cases of phase naming.
            phase = 'IM' if j.strip().upper() in ['LM', 'IM'] else j.strip()
            phases.append(phase)
        # Reconstruct the phase string.
        phase_string = '+'.join(phases)

    cleaned_phase_list.append(phase_string if isinstance(phase_list[i], str) else phase_list[i])

phase_list = cleaned_phase_list

# Create a cleaned dataset and save it to a new Excel file.
output_df = pandas.DataFrame(
    list(zip(alloy_list, phase_list, hardness_list, yield_list, tensile_list, elongation_list, compressive_list,
             plasticity_list, process_list, source_list)), columns=feature[1:11]).sort_values(feature[1])

output_df.to_excel('MPEA_cleaned_dataset.xlsx', index=False)
