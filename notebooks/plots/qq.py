import pandas as pd
import os
import glob
from IPython.display import display, Latex

data_list = ['christensen', 'sun', 'alon', 'nakayama', 'tian', 'chin', 'chowdary', 
             'subramanian', 'chiaretti', 'golub', 'shipp', 'su', 'gordon', 'khan', 
             'singh', 'gravier', 'borovecki', 'sorlie', 'west']

PATH_TEST = '/Users/erirosv/fun/Master-Thesis/plotting-result/version2/merged-data'
p = os.path.abspath(PATH_TEST)

# Reading the data from the files
datasets = []
for d in data_list:
    csv_files = glob.glob(os.path.join(p, f'results_{d}.csv'))
    for csv_file in csv_files:
        dataset = pd.read_csv(csv_file)
        dataset['classification_accuracy'] = 100 * (1 - pd.to_numeric(dataset['cv_error_mean'], errors='coerce'))
        datasets.append(dataset)

print(datasets[1]['fs_method'].unique())
print(datasets[1]['dataset'].unique())

complete_dataset_avg = datasets.groupby(['fs_method', '_wrapper'])['classification_accuracy'].mean().reset_index()

# Pivot the data and format values with two decimal places
complete_dataset_avg_table = pd.pivot_table(complete_dataset_avg, values='classification_accuracy', index='_wrapper',
                                            columns='fs_method')

# Calculate the row-wise average and add to the table
complete_dataset_avg_table.loc['Average'] = complete_dataset_avg_table.mean()

# Convert the table to LaTeX format with two decimal places
latex_table = complete_dataset_avg_table.to_latex(float_format='%.2f')

# Display the LaTeX table
display(Latex(latex_table))
print(latex_table)
