import pandas

dataset = pandas.read_excel('MPEA_mechanical_database.xlsx')
feature = dataset.columns.values

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

# remove spaces at the beginning and at the end of each string.
comment_lists = [process_list, source_list]

for comment_list in comment_lists:
    for i in range(len(comment_list)):
        if isinstance(comment_list[i], str):
            comment_list[i] = comment_list[i].strip()

for i in range(len(alloy_list)):
    alloy_list[i] = alloy_list[i].replace(" ", "")

# for the phase strings, clean the phase formats, and create a cleaned phase list
cleaned_phase_list = []
for i in range(len(phase_list)):
    phase_string = ""
    # if the entry is a string, parse the phases and reconstruct the string
    if isinstance(phase_list[i], str):
        phases = []
        split_list = phase_list[i].split('+')
        for j in range(len(split_list)):
            # correct the mistakenly recorded 'LM' phase
            if split_list[j].strip().upper() == 'LM' or split_list[j].strip().upper() == 'IM':
                phases.append('IM')
            else:
                phases.append(split_list[j].strip())
        # reconstruct the string
        for k in range(len(phases)):
            if k > 0:
                phase_string += '+'
            phase_string += phases[k]

    if isinstance(phase_list[i], str):
        cleaned_phase_list.append(phase_string)
    else:
        cleaned_phase_list.append(phase_list[i])

phase_list = cleaned_phase_list

# rewrite a new dataset that contains only the un-parsed information of the dataset
output_df = pandas.DataFrame(
    list(zip(alloy_list, phase_list, hardness_list, yield_list, tensile_list, elongation_list, compressive_list,
             plasticity_list, process_list, source_list)), columns=feature[1:11]).sort_values(feature[1])

output_df.to_excel('MPEA_cleaned_dataset.xlsx', index=False)

