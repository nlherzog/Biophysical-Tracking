import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def run_anova_and_tukey(input_csv, output_dir, identifier_str, alpha=0.02, bonferroni_correction=4): #change bonferroni correction to number of groups in comparison
    # Load data
    df = pd.read_csv(input_csv)

    # Check required columns
    for col in ['aexp', 'log_D', 'group']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    corrected_alpha = alpha / bonferroni_correction

   # ---- log_D stats ----
    model_logD = smf.ols('log_D ~ C(group)', data=df).fit()
    anova_logD = sm.stats.anova_lm(model_logD, typ=2)
    anova_logD.to_csv(f"{output_dir}/{identifier_str}_anova_logD.csv", index=False)

    tukey_logD = pairwise_tukeyhsd(df['log_D'], df['group'], alpha=corrected_alpha)
    with open(f"{output_dir}/{identifier_str}_tukey_logD.txt", "w") as f:
        f.write(str(tukey_logD))

    # ---- aexp stats ----
    model_aexp = smf.ols('aexp ~ C(group)', data=df).fit()
    anova_aexp = sm.stats.anova_lm(model_aexp, typ=2)
    anova_aexp.to_csv(f"{output_dir}/{identifier_str}_anova_aexp.csv", index=False)

    tukey_aexp = pairwise_tukeyhsd(df['aexp'], df['group'], alpha=corrected_alpha)
    with open(f"{output_dir}/{identifier_str}_tukey_aexp.txt", "w") as f:
        f.write(str(tukey_aexp))

    print("Statistical results saved in:", output_dir)

if __name__ == "__main__":
    input_csv = "PATH-TO-INPUT-CSV"  # This is the specific "{identifier_str}_alldata_modified.csv"
    output_dir = "PATH-TO-DIRECTORY-WITH-FILES"
    identifier_str = "EXPERIMENT-ID-STRINGS" #identifier string

run_anova_and_tukey(input_csv, output_dir, identifier_str)
