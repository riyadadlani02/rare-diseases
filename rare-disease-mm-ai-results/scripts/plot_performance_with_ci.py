"""
This script generates a grouped bar chart to visualize diagnostic performance metrics
for different rare diseases, including asymmetric 95% confidence intervals.

The script performs the following steps:
1.  **Data Loading**: It reads a CSV file named `diagnostic_performance_ci.csv`,
    which is expected to contain columns for 'Disease', 'Metric', 'Mean (%)',
    '95% CI Lower (%)', and '95% CI Upper (%)'.

2.  **Data Processing**:
    -   It defines the diseases and metrics of interest ('Huntington’s', 'DMD', 'Gaucher'
      and 'Accuracy', 'Sensitivity', 'Specificity').
    -   It pivots the data to create separate dictionaries for mean performance values
      and the lower and upper bounds of the confidence intervals for each disease.
    -   The confidence intervals are calculated as the difference from the mean, which
      is required for `matplotlib`'s error bar functionality.

3.  **Plotting**:
    -   It uses `matplotlib` to create a grouped bar chart. Each group of bars
      represents a disease.
    -   Within each group, there is one bar for each metric (Accuracy, Sensitivity,
      and Specificity).
    -   Asymmetric error bars are added to each bar to represent the 95% confidence
      intervals.
    -   The plot is customized with labels, a title, and a legend.

4.  **Saving the Output**:
    -   The final plot is saved as a high-resolution PNG image to the `figures`
      directory.

To run the script, ensure the input CSV file is in the `data/` directory relative
to where the script is executed.

Example CSV format:
Disease,Metric,Mean (%),95% CI Lower (%),95% CI Upper (%)
Huntington’s,Accuracy,95.5,94.2,96.8
...
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data/diagnostic_performance_ci.csv')
    diseases = ['Huntington’s', 'DMD', 'Gaucher']
    metrics = ['Accuracy', 'Sensitivity', 'Specificity']

    # pivot to get means & CIs
    means = {d: [] for d in diseases}
    lowers = {d: [] for d in diseases}
    uppers = {d: [] for d in diseases}
    for d in diseases:
        for m in metrics:
            row = df[(df['Disease']==d) & (df['Metric']==m)].iloc[0]
            means[d].append(row['Mean (%)'])
            lowers[d].append(row['Mean (%)'] - row['95% CI Lower (%)'])
            uppers[d].append(row['95% CI Upper (%)'] - row['Mean (%)'])

    bar_width = 0.25
    x = np.arange(len(diseases))
    offsets = [-bar_width, 0, bar_width]

    plt.figure(figsize=(12,6))
    for i, m in enumerate(metrics):
        m_means = [means[d][i] for d in diseases]
        m_lowers = [lowers[d][i] for d in diseases]
        m_uppers = [uppers[d][i] for d in diseases]
        plt.bar(x + offsets[i], m_means, bar_width,
                yerr=[m_lowers, m_uppers], capsize=6, label=m)

    plt.ylabel('Percentage (%)')
    plt.title('Diagnostic Performance by Disease with Asymmetric 95% CIs')
    plt.xticks(x, diseases)
    plt.ylim(80, 100)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('figures/diagnostic_performance_asymmetric_ci.png', dpi=300, bbox_inches='tight')
    print('Saved to figures/diagnostic_performance_asymmetric_ci.png')

if __name__ == '__main__':
    main()
