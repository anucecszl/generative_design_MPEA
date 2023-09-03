import pandas
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PM
from joblib import load
import WGAN_GP
import torch

dataset_df = pandas.read_excel('../dataset/MPEA_parsed_dataset.xlsx')
element_names = dataset_df.columns.values[14:46]
process_names = dataset_df.columns.values[46:53]
feature_names = dataset_df.columns.values[14:53]

data_np = dataset_df.to_numpy()
# identify the features of the alloys.
comp_data = data_np[:, 14:53].astype(float)
comp_min = np.min(comp_data, axis=0)
comp_max = np.max(comp_data, axis=0)


generator = WGAN_GP.Generator()
generator.load_state_dict(torch.load('../saved_models/generator_net_MPEA.pt'))
generator.eval()

hard_regressor = load('../saved_models/hardness_regressor.joblib')
yield_regressor = load('../saved_models/yield_regressor.joblib')
tensile_regressor = load('../saved_models/tensile_regressor.joblib')
elongation_regressor = load('../saved_models/elongation_regressor.joblib')


class AlloyOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=10, n_obj=3, xl=-3, xu=3)

    def _evaluate(self, x, out, *args, **kwargs):
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            fake_alloys = generator(x_tensor).numpy()

        fake_alloys = fake_alloys * comp_max + comp_min

        f1 = - tensile_regressor.predict(fake_alloys)
        f2 = - hard_regressor.predict(fake_alloys)
        f3 = - fake_alloys[:, 8]
        out["F"] = np.column_stack([f1, f2, f3])


problem = AlloyOptimizationProblem()
algorithm = NSGA2(pop_size=100, mutation=PM(prob=0.1, eta=20))
termination = get_termination("n_gen", 200)

res = minimize(problem,
               algorithm,
               termination,
               pf=problem.pareto_front(), save_history=True,
               seed=3,
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

formula_df = pandas.DataFrame(zip(alloy_names, process_name_list),
                              columns=['Composition', 'Processing method'])

property_df = pandas.DataFrame(property_array, columns=['Elongation', 'Tensile', 'Yield', 'Hardness'])
feature_df = pandas.DataFrame(optimal_alloys, columns=list(feature_names))
# convert the data into pandas dataframe and output
output_df = pandas.concat([formula_df, property_df, feature_df], axis=1)

output_df.to_excel('../generated_datasets/generated_NSGAN.xlsx', index=False)

gene_df = pandas.DataFrame(res.X,
                           columns=['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6', 'gene7', 'gene8', 'gene9',
                                    'gene10', ])
gene_mapping_df = pandas.concat([gene_df, formula_df, property_df], axis=1)
gene_mapping_df.to_excel('../generated_datasets/NSGAN_mapping.xlsx', index=False)
