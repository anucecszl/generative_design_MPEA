import pandas as pd
import numpy as np
import pymoo
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from joblib import load
from torch import nn
import torch
import streamlit as st
import matplotlib.pyplot as plt
import base64

# Define process name to description mapping
process_name_mapping = {
    'process_1': "As-cast processes, inclusive of 'arc-melted'",
    'process_2': "Arc-melted processes followed by artificial aging",
    'process_3': "Arc-melted processes followed by annealing",
    'process_4': "Powder processing techniques (powder metallurgy)",
    'process_5': "Novel synthesis techniques (i.e., ball milling,)",
    'process_6': "Arc-melted processes followed by wrought processing techniques",
    'process_7': "Cryogenic treatments"
}


# Define the Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
        )

    def forward(self, noise):
        fake_formula = self.model(noise)
        return fake_formula


class AlloyOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=2, xl=-3, xu=3)

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            fake_alloys = generator(x_tensor).numpy()
        fake_alloys = fake_alloys * comp_max + comp_min
        f1 = - tensile_regressor.predict(fake_alloys)
        f2 = - elongation_regressor.predict(fake_alloys)
        out["F"] = np.column_stack([f1, f2])


st.markdown("<h3 style='text-align: center; color: black;'>Multi Principle Element Alloy Generation and Property Prediction</h3>",
            unsafe_allow_html=True)
st.markdown("""
This online tool employs the NSGAN model (non-dominant sorting optimization-based generative adversarial network) 
to generate optimized element compositions, processing conditions, 
and predicted phase and mechanical properties (including hardness, tensile strength, yield strength, elongation) for 
multi-principle element alloys. 

The "Population Size" specifies the number of candidate solutions explored in each 
generation and impacts the diversity of solutions; the "Number of Generations" indicates how many iterative cycles 
the algorithm will undergo, influencing the quality and convergence of solutions; and the "Random Seed Value" is 
utilized to initialize the random number generator, allowing for reproducibility in the optimization process. The 
ArXiv paper can be access at: 'link'
""")
# Create a layout with three columns
cols = st.columns(3)

# Assign each input to a different column
pop_size = cols[0].number_input("Population Size", min_value=10, max_value=50, value=20)
n_gen = cols[1].number_input("Number of Generations", min_value=5, max_value=500, value=100)
seed_value = cols[2].number_input("Random Seed Value", min_value=0, max_value=None, value=2)

start_optimization = st.button("Start Optimization")

# 检查session_state中是否存在结果DataFrame
if "result_df" not in st.session_state:
    st.session_state.result_df = None
    
if start_optimization:
    with st.spinner('Optimizing...'):
        # Load data and models
        element_names = pd.read_excel('streamlit/MPEA_parsed_dataset.xlsx').columns.values[
                        14:46]
        process_names = pd.read_excel('streamlit/MPEA_parsed_dataset.xlsx').columns.values[
                        46:53]
        dataset_df = pd.read_excel('streamlit/MPEA_parsed_dataset.xlsx')
        data_np = dataset_df.to_numpy()

        comp_data = data_np[:, 14:53].astype(float)
        comp_min = np.min(comp_data, axis=0)
        comp_max = np.max(comp_data, axis=0)
        feature_names = dataset_df.columns.values[14:53]

        generator = Generator()
        generator.load_state_dict(torch.load('streamlit/generator_net_MPEA.pt'))
        generator.eval()

        yield_regressor = load('streamlit/yield_regressor.joblib')
        tensile_regressor = load('streamlit/tensile_regressor.joblib')
        elongation_regressor = load('streamlit/elongation_regressor.joblib')
        hard_regressor = load('streamlit/hardness_regressor.joblib')

        # Optimize
        problem = AlloyOptimizationProblem()
        algorithm = NSGA2(pop_size=pop_size, mutation=PM(prob=0.1, eta=20))
        termination = get_termination("n_gen", n_gen)

        res = minimize(problem, algorithm, termination, pf=problem.pareto_front(), save_history=True, seed=seed_value,
                       verbose=False)

        result_tensor = torch.tensor(res.X, dtype=torch.float32)
        with torch.no_grad():
            optimal_alloys = generator(result_tensor).numpy()

        optimal_alloys = optimal_alloys * comp_max + comp_min
        optimal_alloys[:, :32] = optimal_alloys[:, :32] / np.sum(optimal_alloys[:, :32], axis=1).reshape((-1, 1))

        # create a list for the generated alloys' name
        alloy_names = []
        for i in range(optimal_alloys.shape[0]):
            composition = optimal_alloys[i, :32]
            comp_string = ''
            element_list = []
            ratio_list = []
            for j in range(len(composition)):
                # only keep the molar ratio > 0.005
                if composition[j] > 0.005:
                    comp_string += element_names[j]
                    comp_string += str(round(composition[j], 3))
                    element_list.append(element_names[j])
                    ratio_list.append(composition[j])
                else:
                    composition[j] = 0
            alloy_names.append(comp_string)

        process_indices = np.argmax(optimal_alloys[:, 32:], axis=1)
        process_one_hot = np.zeros_like(optimal_alloys[:, 32:])
        for i in range(len(process_one_hot)):
            for j in range(len(process_one_hot[i])):
                if process_indices[i] == j:
                    process_one_hot[i][j] = 1
        optimal_alloys[:, 32:] = process_one_hot

        property_array = np.zeros((optimal_alloys.shape[0], 4))
        property_array[:, 0] = elongation_regressor.predict(optimal_alloys)
        property_array[:, 1] = tensile_regressor.predict(optimal_alloys)
        property_array[:, 2] = yield_regressor.predict(optimal_alloys)
        property_array[:, 3] = hard_regressor.predict(optimal_alloys)

        process_name_list = []
        for i in range(len(optimal_alloys)):
            process_name_list.append(process_names[process_indices[i]])

        # Predict phases and mechanical properties for generated alloys
        phase_array = np.zeros((optimal_alloys.shape[0], 4))
        classifier_FCC = load('streamlit/FCC_classifier.joblib')
        classifier_BCC = load('streamlit/BCC_classifier.joblib')
        classifier_HCP = load('streamlit/HCP_classifier.joblib')
        classifier_IM = load('streamlit/IM_classifier.joblib')

        phase_array[:, 0] = classifier_FCC.predict(optimal_alloys)
        phase_array[:, 1] = classifier_BCC.predict(optimal_alloys)
        phase_array[:, 2] = classifier_HCP.predict(optimal_alloys)
        phase_array[:, 3] = classifier_IM.predict(optimal_alloys)

        print('Finished phase prediction.')

        # Identify the predicted phases as strings
        phase_name_list = []
        phase_name_strings = ['FCC', 'BCC', 'HCP', 'IM']
        for i in range(optimal_alloys.shape[0]):
            phase_name = ''
            for j in range(4):
                if phase_array[i][j] > 0:
                    phase_name += phase_name_strings[j]
                    phase_name += '+'
            phase_name_list.append(phase_name[:-1])

        # Plot Tensile vs Elongation
        st.subheader("Tensile strength vs Elongation:")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(property_array[:, 1], property_array[:, 0])
        ax1.set_xlabel("Tensile strength (MPa)")
        ax1.set_ylabel("Elongation (%)")
        ax1.set_title("Tensile strength vs Elongation")
        st.pyplot(fig1)

        # Plot Yield vs Elongation
        st.subheader("Yield strength vs Elongation:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(property_array[:, 2], property_array[:, 0])
        ax2.set_xlabel("Yield strength (MPa)")
        ax2.set_ylabel("Elongation (%)")
        ax2.set_title("Yield strength vs Elongation")
        st.pyplot(fig2)

        # save the result to DataFrame
        result_df = pd.DataFrame({
            'Alloy Composition': alloy_names,
            'Processing Method': [process_name_mapping.get(process_name_list[i], "Unknown") for i in
                                  range(len(alloy_names))],
            'Predicted Phase': phase_name_list,
            'Hardness HV': property_array[:, 3],
            'Tensile Strength MPa': property_array[:, 1],
            'Yield Strength MPa': property_array[:, 2],
            'Elongation %': property_array[:, 0]
        })

        # create the download button
        if st.button("Download Results as Excel") and st.session_state.result_df is not None:
            output_path = "optimization_results.xlsx"
            st.session_state.result_df.to_excel(output_path, index=False, engine='openpyxl')

            with open(output_path, "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:file/xlsx;base64,{b64}" download="{output_path}">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Display the results
        st.subheader("Optimal Alloys:")
        for i in range(len(alloy_names)):
            st.markdown(f"**Alloy {i + 1}:**")
            st.write(f"Composition: {alloy_names[i]}")
            # Mapping the process name to its description
            detailed_process_name = process_name_mapping.get(process_name_list[i], "Unknown")
            st.write(f"Process Method: {detailed_process_name}")
            st.write(
                f"Predicted phase: {phase_name_list[i]}")
            st.write(
                f"Hardness: {property_array[i][3]:.2f} HV, Tensile strength: {property_array[i][1]:.2f} MPa, Yield strength: {property_array[i][2]:.2f} MPa, Elongation: {property_array[i][0]:.2f} %")

