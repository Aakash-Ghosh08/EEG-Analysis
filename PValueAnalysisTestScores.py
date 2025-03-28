import pandas as pd
import numpy as np
from scipy import stats

# Create DataFrame
data = {
    'Participant': [1, 2, 3, 4, 5, 6, 7, 8],
    'Group': ['Experimental', 'Experimental', 'Experimental', 'Control', 'Control', 'Experimental', 'Experimental', 'Control'],
    'MoCA Total': [28.5, 32.0, 30.5, 30.5, 30.5, 29.0, 32.0, 27.0],
    'MMSE Total': [28, 30, 30, 28, 29, 29, 30, 27],
    'Mini-Cog Total': [5, 5, 2, 3, 4, 5, 5, 5]
}

df = pd.DataFrame(data)

# Calculate combined score without normalization
df['Combined Score'] = df['MoCA Total'] + df['MMSE Total'] + df['Mini-Cog Total']

# Separate experimental and control groups
experimental = df[df['Group'] == 'Experimental']
control = df[df['Group'] == 'Control']

# Perform t-test on combined scores
t_statistic, p_value = stats.ttest_ind(
    experimental['Combined Score'], 
    control['Combined Score']
)

print("Combined Test Score Analysis:")
print("\nDetailed Breakdown:")
print("Experimental Group:")
for i, row in experimental.iterrows():
    print(f"Participant {row['Participant']}: MoCA {row['MoCA Total']}, MMSE {row['MMSE Total']}, Mini-Cog {row['Mini-Cog Total']}, Combined {row['Combined Score']}")

print("\nControl Group:")
for i, row in control.iterrows():
    print(f"Participant {row['Participant']}: MoCA {row['MoCA Total']}, MMSE {row['MMSE Total']}, Mini-Cog {row['Mini-Cog Total']}, Combined {row['Combined Score']}")

print("\nStatistical Summary:")
print(f"Experimental Group Combined Score Mean: {experimental['Combined Score'].mean():.4f}")
print(f"Control Group Combined Score Mean: {control['Combined Score'].mean():.4f}")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Check for statistical significance
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant Difference")
else:
    print("Result: No Statistically Significant Difference")