import pandas
import numpy
from pymatgen.core.periodic_table import Element
import empirical_parameter_calculator as calculator

# Import mechanical property and mapping data from Excel files
mech_data = pandas.read_excel('MPEA_cleaned_dataset.xlsx')
mapping_data = pandas.read_excel('MPEA_process_method_map.xlsx')
element_set = set()


def parse(comp):
    """Decode elemental composition and extract elemental molar ratios.

    Parameters:
        comp (str): The elemental composition string.

    Returns:
        ele_list (list): List of elements.
        ratio_list (list): List of corresponding molar ratios.
    """

    ele_list = []
    ratio_list = []

    head = 0
    while head < len(comp):
        if not comp[head].isupper():
            head += 1
            continue

        num_start = head + 1
        num_end = num_start
        if head + 1 < len(comp) and comp[head + 1].islower():
            ele = comp[head:head + 2]
            num_start = head + 2
        else:
            ele = comp[head:head + 1]

        for i in range(len(comp) - num_start):
            if not comp[num_start + i].isnumeric() and not comp[num_start + i] == '.':
                num_end = num_start + i
                break
            if num_start + i == len(comp) - 1:
                num_end = num_start + i + 1
        number = comp[num_start:num_end]
        ele_list.append(ele.strip())
        element_set.add(ele.strip())

        ele_ratio = float(1)
        if len(number.strip()) > 0:
            ele_ratio = float(number.strip())
        ratio_list.append(ele_ratio)
        head += 1
    return ele_list, ratio_list


def decode(alloy_name):
    """Recursive function to decode alloy compositions, handling special cases with parentheses.

    Parameters:
        alloy_name (str): The name of the alloy.

    Returns:
        ele_list (list): List of elements.
        ratio_list (list): List of corresponding molar ratios.
    """
    comp = alloy_name.strip()
    paren_start = 0
    paren_end = 0
    has_paren = False

    # Check if there is a parenthesis in the composition, record the position of parenthesis.
    for k in range(len(comp)):
        if comp[k] == '(':
            paren_start = k
            has_paren = True
        if comp[k] == ')':
            paren_end = k
    # If there is, parse the composition, molar ratio of the elements in the parenthesis.
    if has_paren:
        comp_in_paren = comp[paren_start + 1:paren_end]
        paren_num = ""
        paren_num_end = 0
        for k in range(len(comp) - paren_end - 1):
            if k > len(comp) - 1 or comp[k + 1 + paren_end].isupper():
                break
            paren_num = comp[paren_end + 1:k + 2 + paren_end]
            paren_num_end = k + 2 + paren_end
        # Decode the compositions before and after the parenthesis.
        comp_front = comp[0:paren_start]
        comp_end = comp[paren_num_end:]

        paren_ele_list, paren_ratio_list = parse(comp_in_paren)
        paren_ratio = float(paren_num) / sum(paren_ratio_list)
        for k in range(len(paren_ratio_list)):
            paren_ratio_list[k] = round(paren_ratio_list[k] * paren_ratio, 2)

        ele_list = paren_ele_list
        ratio_list = paren_ratio_list

        if len(comp_front) > 0:
            front_ele_list, front_ratio_list = decode(comp_front)
            ele_list = front_ele_list + ele_list
            ratio_list = front_ratio_list + ratio_list
        if len(comp_end) > 0:
            end_ele_list, end_ratio_list = decode(comp_end)
            ele_list = ele_list + end_ele_list
            ratio_list = ratio_list + end_ratio_list

    else:
        ele_list, ratio_list = parse(comp)

    return ele_list, ratio_list


def normalize(decoded_result):
    """Normalize the elemental molar ratios.

    Parameters:
        decoded_result (tuple): A tuple containing decoded elements and corresponding molar ratios.

    Returns:
        normalized_elements (list): List of normalized elements.
        normalized_ratios (list): List of normalized molar ratios.
    """
    decoded_elements, decoded_ratios = decoded_result
    normalized_elements = []
    normalized_ratios = []
    for k in range(len(decoded_elements)):
        if decoded_elements[k] in normalized_elements:
            ind = normalized_elements.index(decoded_elements[k])
            normalized_ratios[ind] += decoded_ratios[k]
        else:
            normalized_elements.append(decoded_elements[k])
            normalized_ratios.append(decoded_ratios[k])

    ratios_sum = sum(normalized_ratios)
    for k in range(len(normalized_ratios)):
        normalized_ratios[k] = round(normalized_ratios[k] / ratios_sum, 3)

    return normalized_elements, normalized_ratios


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Extract features from the imported mechanical property data
    feature = mech_data.columns.values
    # Additional variables like phase_list, hardness_list, etc.
    alloy_list = mech_data[feature[0]]
    phase_list = mech_data[feature[1]]
    hardness_list = mech_data[feature[2]]
    yield_list = mech_data[feature[3]]
    tensile_list = mech_data[feature[4]]
    elongation_list = mech_data[feature[5]]
    compressive_list = mech_data[feature[6]]
    plasticity_list = mech_data[feature[7]]
    process_list = mech_data[feature[8]]
    refer_list = mech_data[feature[9]]

    decoded_alloy_list = []
    decoded_ratio_list = []

    # For each alloy composition in the dataset, decode and parse each composition and add the decoded result into list
    for i in range(len(alloy_list)):
        alloy_comp = alloy_list[i]
        decoded_alloy, decoded_ratio = normalize(decode(alloy_comp))
        decoded_alloy_list.append(decoded_alloy)
        decoded_ratio_list.append(decoded_ratio)

    # Count all the element occurred in the decoding and create an element list.
    element_list = sorted(list(element_set))

    # ELEMENT MOLAR RATIO ARRAY: use a numpy array to record all the molar ratios for the whole dataset
    n_y = len(mech_data[feature[0]])
    n_x = len(element_list)
    ele_ratio_array = numpy.zeros((n_y, n_x))
    for i in range(n_y):
        decoded_alloy = decoded_alloy_list[i]
        decoded_ratio = decoded_ratio_list[i]
        for j in range(len(decoded_alloy)):
            element = decoded_alloy[j]
            ratio = decoded_ratio[j]
            index = element_list.index(element)
            ele_ratio_array[i][index] += ratio

    # PHASE ARRAY: for each alloy composition, decode its phase information into a list of [FCC, BCC, HCP, IM]
    phase_array = numpy.zeros((n_y, 4))

    for i in range(len(phase_list)):
        if isinstance(phase_list[i], str):
            phases = []
            split_list = phase_list[i].split('+')
            for j in range(len(split_list)):
                phases.append(split_list[j].strip().upper())

            for phase in phases:
                if 'FCC' in phase:
                    phase_array[i][0] = 1
                elif 'BCC' in phase:
                    phase_array[i][1] = 1
                elif 'HCP' in phase:
                    phase_array[i][2] = 1
                else:
                    phase_array[i][3] = 1

    # PROCESSING METHOD ENCODING
    process_array = numpy.zeros((n_y, 7))
    mapping_feature = mapping_data.columns.values

    map_process_list = list(mapping_data[mapping_feature[3]])
    for i in range(len(map_process_list)):
        if isinstance(map_process_list[i], str):
            map_process_list[i] = map_process_list[i].strip()

    map_encode_list = list(mapping_data[mapping_feature[4]])

    for i in range(len(process_list)):
        index = map_process_list.index(process_list[i])
        code = map_encode_list[index]
        process_array[i][code - 1] += 1

    # EMPIRICAL PARAMETER CALCULATION
    empirical_array = numpy.zeros((n_y, 15))

    for i in range(len(decoded_alloy_list)):
        print('calculating parameters for the ', i + 1, ' alloy.')
        alloy_elements = []
        mol_ratios = decoded_ratio_list[i]
        for j in range(len(decoded_alloy_list[i])):
            alloy_elements.append(Element(decoded_alloy_list[i][j]))
        alloy = calculator.EmpiricalParams(element_list=alloy_elements, mol_ratio=mol_ratios)
        parameter_list = alloy.get_15_parameters()
        for j in range(len(parameter_list)):
            empirical_array[i][j] = parameter_list[j]

    # Concatenate the parsed result and the original dataset, write to an excel file
    ratios_df = pandas.DataFrame(ele_ratio_array, columns=element_list)
    phases_df = pandas.DataFrame(phase_array, columns=['FCC', 'BCC', 'HCP', 'IM'])
    process_df = pandas.DataFrame(process_array,
                                  columns=['process_1', 'process_2', 'process_3', 'process_4', 'process_5', 'process_6',
                                           'process_7'])
    empirical_df = pandas.DataFrame(empirical_array,
                                    columns=['a', 'delta', 'Tm', 'std of Tm', 'entropy', 'enthalpy', 'std of enthalpy',
                                             'omega', 'X', 'std of X', 'VEC', 'std of vec', 'K', 'std of K', 'density'])

    output_df = pandas.concat([mech_data, phases_df, ratios_df, process_df, empirical_df], axis=1)
    output_df.to_excel('MPEA_parsed_dataset.xlsx', index=False)
