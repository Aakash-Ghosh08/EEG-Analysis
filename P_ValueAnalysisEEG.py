import pandas as pd
import numpy as np
from scipy import stats

# Define the groups
control_indices = [4, 5, 8]
experimental_indices = [1, 2, 3, 6, 7]  # Note: 6 is the average of 6_1 and 6_2

# Control Group data
control_data = {
    "4": {
        "Mean": -2.2870,
        "Std_Dev": 228.1723,
        "Variance": 52062.6152,
        "Skewness": -0.1722,
        "Kurtosis": 14.3316,
        "Range": 2713.0062,
        "Interquartile_Range": 98.2842,
        "Sample_Entropy": 0.5799,
        "Perm_Entropy": 0.9548,
        "Hjorth_Mobility": 0.6995,
        "Hjorth_Complexity": 1.9793,
        "Delta_Abs": 18124.7643,
        "Theta_Abs": 2390.3398,
        "Alpha_Abs": 809.7490,
        "Beta_Abs": 1166.8343,
        "Gamma_Abs": 485.0674,
        "Delta_Rel": 0.7888,
        "Theta_Rel": 0.1040,
        "Alpha_Rel": 0.0352,
        "Beta_Rel": 0.0508,
        "Gamma_Rel": 0.0211
    },
    "5": {
        "Mean": -7.6703,
        "Std_Dev": 688.1039,
        "Variance": 473487.0123,
        "Skewness": -0.0218,
        "Kurtosis": 4.4938,
        "Range": 6458.3082,
        "Interquartile_Range": 834.8929,
        "Sample_Entropy": 0.7852,
        "Perm_Entropy": 0.9563,
        "Hjorth_Mobility": 0.8392,
        "Hjorth_Complexity": 1.6255,
        "Delta_Abs": 154594.7976,
        "Theta_Abs": 14871.1828,
        "Alpha_Abs": 3779.6732,
        "Beta_Abs": 3091.7113,
        "Gamma_Abs": 529.0684,
        "Delta_Rel": 0.8741,
        "Theta_Rel": 0.0841,
        "Alpha_Rel": 0.0214,
        "Beta_Rel": 0.0175,
        "Gamma_Rel": 0.0030
    },
    "8": {
        "Mean": 0.5110,
        "Std_Dev": 50.4503,
        "Variance": 2545.2356,
        "Skewness": -0.0626,
        "Kurtosis": 3.9613,
        "Range": 414.4989,
        "Interquartile_Range": 42.5431,
        "Sample_Entropy": 1.1084,
        "Perm_Entropy": 0.9548,
        "Hjorth_Mobility": 0.4284,
        "Hjorth_Complexity": 3.0858,
        "Delta_Abs": 1075.6277,
        "Theta_Abs": 90.8239,
        "Alpha_Abs": 29.7825,
        "Beta_Abs": 49.1519,
        "Gamma_Abs": 28.1130,
        "Delta_Rel": 0.8446,
        "Theta_Rel": 0.0713,
        "Alpha_Rel": 0.0234,
        "Beta_Rel": 0.0386,
        "Gamma_Rel": 0.0221
    }
}

# Experimental Group data
experimental_data = {
    "1": {
        "Mean": 0.7898,
        "Std_Dev": 194.4336,
        "Variance": 37804.4400,
        "Skewness": -0.1010,
        "Kurtosis": 8.9838,
        "Range": 2031.9654,
        "Interquartile_Range": 87.6620,
        "Sample_Entropy": 0.3428,
        "Perm_Entropy": 0.9301,
        "Hjorth_Mobility": 0.1964,
        "Hjorth_Complexity": 5.3433,
        "Delta_Abs": 17489.4825,
        "Theta_Abs": 2818.9386,
        "Alpha_Abs": 1069.5937,
        "Beta_Abs": 1423.4640,
        "Gamma_Abs": 273.9819,
        "Delta_Rel": 0.7579,
        "Theta_Rel": 0.1222,
        "Alpha_Rel": 0.0464,
        "Beta_Rel": 0.0617,
        "Gamma_Rel": 0.0119
    },
    "2": {
        "Mean": -0.1812,
        "Std_Dev": 67.8044,
        "Variance": 4597.4364,
        "Skewness": -0.4589,
        "Kurtosis": 53.5298,
        "Range": 1352.4123,
        "Interquartile_Range": 19.4132,
        "Sample_Entropy": 0.1588,
        "Perm_Entropy": 0.8738,
        "Hjorth_Mobility": 0.1002,
        "Hjorth_Complexity": 9.8778,
        "Delta_Abs": 1671.6352,
        "Theta_Abs": 76.4770,
        "Alpha_Abs": 28.4181,
        "Beta_Abs": 37.0599,
        "Gamma_Abs": 16.7146,
        "Delta_Rel": 0.9133,
        "Theta_Rel": 0.0418,
        "Alpha_Rel": 0.0155,
        "Beta_Rel": 0.0202,
        "Gamma_Rel": 0.0091
    },
    "3": {
        "Mean": 6.2523,
        "Std_Dev": 805.2576,
        "Variance": 648439.7266,
        "Skewness": 2.9663,
        "Kurtosis": 279.9562,
        "Range": 33428.9624,
        "Interquartile_Range": 33.5353,
        "Sample_Entropy": 0.0082,
        "Perm_Entropy": 0.8400,
        "Hjorth_Mobility": 0.1034,
        "Hjorth_Complexity": 11.3859,
        "Delta_Abs": 257421.1511,
        "Theta_Abs": 11675.0148,
        "Alpha_Abs": 2905.3230,
        "Beta_Abs": 4827.4758,
        "Gamma_Abs": 1659.1824,
        "Delta_Rel": 0.9244,
        "Theta_Rel": 0.0419,
        "Alpha_Rel": 0.0104,
        "Beta_Rel": 0.0173,
        "Gamma_Rel": 0.0060
    },
    "6": {  # Average of 6_1 and 6_2
        "Mean": (0.7831 + 0.4549) / 2,
        "Std_Dev": (58.8496 + 56.8813) / 2,
        "Variance": (3463.2764 + 3235.4831) / 2,
        "Skewness": (-0.2029 + 0.0466) / 2,
        "Kurtosis": (5.2539 + 9.2285) / 2,
        "Range": (508.7178 + 580.9684) / 2,
        "Interquartile_Range": (43.7596 + 27.9845) / 2,
        "Sample_Entropy": (0.4780 + 0.2576) / 2,
        "Perm_Entropy": (0.9444 + 0.9197) / 2,
        "Hjorth_Mobility": (0.1451 + 0.1072) / 2,
        "Hjorth_Complexity": (8.2785 + 9.4886) / 2,
        "Delta_Abs": (1495.4267 + 1309.6152) / 2,
        "Theta_Abs": (66.3353 + 62.0269) / 2,
        "Alpha_Abs": (17.1571 + 16.2132) / 2,
        "Beta_Abs": (21.5713 + 22.7393) / 2,
        "Gamma_Abs": (10.7721 + 11.9890) / 2,
        "Delta_Rel": (0.9281 + 0.9206) / 2,
        "Theta_Rel": (0.0412 + 0.0436) / 2,
        "Alpha_Rel": (0.0106 + 0.0114) / 2,
        "Beta_Rel": (0.0134 + 0.0160) / 2,
        "Gamma_Rel": (0.0067 + 0.0084) / 2
    },
    "7": {
        "Mean": 0.4441,
        "Std_Dev": 102.6111,
        "Variance": 10529.0306,
        "Skewness": 0.1863,
        "Kurtosis": 11.3247,
        "Range": 1160.6352,
        "Interquartile_Range": 76.5499,
        "Sample_Entropy": 1.0708,
        "Perm_Entropy": 0.9552,
        "Hjorth_Mobility": 0.5230,
        "Hjorth_Complexity": 2.5663,
        "Delta_Abs": 4854.9553,
        "Theta_Abs": 648.5315,
        "Alpha_Abs": 161.6845,
        "Beta_Abs": 182.2303,
        "Gamma_Abs": 56.4724,
        "Delta_Rel": 0.8223,
        "Theta_Rel": 0.1098,
        "Alpha_Rel": 0.0274,
        "Beta_Rel": 0.0309,
        "Gamma_Rel": 0.0096
    }
}

# Extract all metrics from the data
metrics = list(control_data["4"].keys())

# Prepare data for statistical tests
stat_results = {}

for metric in metrics:
    # Extract control group values
    control_values = [data[metric] for data in control_data.values()]
    
    # Extract experimental group values
    experimental_values = [data[metric] for data in experimental_data.values()]
    
    # Perform Mann-Whitney U Test (non-parametric test suitable for small samples)
    u_stat, p_value_mw = stats.mannwhitneyu(control_values, experimental_values, alternative='two-sided')
    
    # For comparison, also calculate t-test (though less reliable with small samples)
    t_stat, p_value_t = stats.ttest_ind(control_values, experimental_values, equal_var=False)
    
    # Store results
    stat_results[metric] = {
        "Mann-Whitney U p-value": p_value_mw,
        "t-test p-value": p_value_t,
        "Control Mean": np.mean(control_values),
        "Experimental Mean": np.mean(experimental_values),
        "Control Values": control_values,
        "Experimental Values": experimental_values
    }

# Create categories for metrics
metrics_groups = {
    "Time-domain features": ["Mean", "Std_Dev", "Variance", "Skewness", "Kurtosis", "Range", "Interquartile_Range"],
    "Complexity features": ["Sample_Entropy", "Perm_Entropy", "Hjorth_Mobility", "Hjorth_Complexity"],
    "Absolute band powers": ["Delta_Abs", "Theta_Abs", "Alpha_Abs", "Beta_Abs", "Gamma_Abs"],
    "Relative band powers": ["Delta_Rel", "Theta_Rel", "Alpha_Rel", "Beta_Rel", "Gamma_Rel"]
}

# Create DataFrames for each category
results_by_category = {}

for category, metrics_in_category in metrics_groups.items():
    category_results = []
    
    for metric in metrics_in_category:
        result = stat_results[metric]
        category_results.append({
            "Metric": metric,
            "Control Mean": result["Control Mean"],
            "Experimental Mean": result["Experimental Mean"],
            "Percent Change": ((result["Experimental Mean"] - result["Control Mean"]) / abs(result["Control Mean"])) * 100 if abs(result["Control Mean"]) > 1e-6 else "N/A",
            "Mann-Whitney p-value": result["Mann-Whitney U p-value"],
            "t-test p-value": result["t-test p-value"],
            "Significant (p<0.05)": "Yes" if result["Mann-Whitney U p-value"] < 0.05 else "No"
        })
    
    results_by_category[category] = pd.DataFrame(category_results)

# Print results for each category
for category, df in results_by_category.items():
    print(f"\n=== {category} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

# Create a summary of significant findings
significant_findings = []

for metric in metrics:
    if stat_results[metric]["Mann-Whitney U p-value"] < 0.05:
        # Find which category this metric belongs to
        category = next((cat for cat, metrics_list in metrics_groups.items() if metric in metrics_list), "Other")
        
        significant_findings.append({
            "Category": category,
            "Metric": metric,
            "Control Mean": stat_results[metric]["Control Mean"],
            "Experimental Mean": stat_results[metric]["Experimental Mean"],
            "Percent Change": ((stat_results[metric]["Experimental Mean"] - stat_results[metric]["Control Mean"]) / abs(stat_results[metric]["Control Mean"])) * 100 if abs(stat_results[metric]["Control Mean"]) > 1e-6 else "N/A",
            "Mann-Whitney p-value": stat_results[metric]["Mann-Whitney U p-value"]
        })

# Print summary of significant findings
print("\n=== Significant Findings (p < 0.05) ===")
if significant_findings:
    significant_df = pd.DataFrame(significant_findings)
    print(significant_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
else:
    print("No statistically significant differences were found.")