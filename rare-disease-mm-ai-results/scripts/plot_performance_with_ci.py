
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
