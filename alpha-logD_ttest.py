import pandas as pd
from scipy.stats import ttest_ind
import os

def run_welchs_ttest(input_csv, output_dir, identifier_str, comparisons, alpha=0.02):
    """
    Perform Welch's t-test on specified group pairs.

    Parameters:
        input_csv (str): Path to the input CSV.
        output_dir (str): Path to save results.
        identifier_str (str): Identifier for output filenames.
        comparisons (list of tuple): List of group name pairs to compare.
        alpha (float): Significance level (used for Bonferroni correction).
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    for col in ['aexp', 'log_D', 'group']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare output
    os.makedirs(output_dir, exist_ok=True)

    # Bonferroni correction
    corrected_alpha = alpha / len(comparisons)

    results = []

    for group1, group2 in comparisons:
        sub1 = df[df['group'] == group1]
        sub2 = df[df['group'] == group2]

        for variable in ['log_D', 'aexp']:
            vals1 = sub1[variable].dropna()
            vals2 = sub2[variable].dropna()

            stat, pval = ttest_ind(vals1, vals2, equal_var=False)

            results.append({
                'variable': variable,
                'group1': group1,
                'group2': group2,
                'mean_group1': vals1.mean(),
                'mean_group2': vals2.mean(),
                't_statistic': stat,
                'p_value': pval,
                'significant': pval < corrected_alpha
            })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"{identifier_str}_welch_ttest_results.csv")
    results_df.to_csv(output_file, index=False)

    print(f"Welch's t-test results saved to: {output_file}")

if __name__ == "__main__":
    input_csv = "PATH-TO-INPUT-CSV"  # This is the specific "{identifier_str}_alldata_modified.csv"
    output_dir = "PATH-TO-DIRECTORY-WITH-FILES"
    identifier_str = "EXPERIMENT-ID-STRINGS" #identifier string

    # Define comparisons: (group_a, group_b), (group_c, group_d), etc.
    comparisons = [
        ('Group1', 'Group2'),
        ('Group3', 'Group4')
    ]

run_welchs_ttest(input_csv, output_dir, identifier_str, comparisons, alpha=0.02)