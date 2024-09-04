import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
# data = {
#     'Challenge Ratio': ['0.25', '0.25', '0.25', '0.5', '0.5', '0.5', '0.75', '0.75', '0.75'],
#     'Incomplete Ratio': [r'P_{TP}@0.2', r'P{TP}@0.8', r'P_{TN}', r'P_{TP}@0.2', r'P{TP}@0.8', r'P_{TN}', r'P_{TP}@0.2', r'P{TP}@0.8', r'P_{TN}'],
#     # 'TPR': [0.64, 0.996, 1.0, 0.88, 1.0, 0.964, 0.988, 1.0, 0.988], # Citeseer
#     # 'TPR': [0.492, 0.948, 0.944, 0.812, 0.996, 0.860, 0.892, 1.0, 0.844], # LastFM
#     'TPR': [0.58, 1, 1, 0.89, 1., 1, 0.96, 1, 0.99] # CS
# }
# df = pd.DataFrame(data)

# df = pd.read_csv('/Users/kunwu/Downloads/mix_verify_unlearning_1724100554_citeseer.csv')
df = pd.read_csv('/Users/kunwu/Downloads/mix_verify_unlearning_1724201712_lastfm.csv')
df = df[df['incomplete_ratio'] != 0.8]
df['x'] = df['num challenge edges'] / 20
df[df['num challenge edges'] == 6]['tpr'] = df[df['num challenge edges'] == 8]
df = df[(df['x'] != 0.2) & (df['x'] != 0.8)]
df = df.melt(id_vars='x', value_vars=['tnr', 'tpr'], var_name='metric', value_name='y')

# Creating the base plot
custom = {
    "axes.edgecolor": "gray", 
    # "grid.linestyle": "dashed", 
    "grid.color": "gray",
} 
sns.set_style("white", rc = custom)
plt.figure()
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=16)
plt.rc('font', size=24)

# Plotting the bars
# ax = sns.barplot(x='x', y='tnr', data=df, hue='metric', palette='muted')
ax = sns.lineplot(x='x', y='y', data=df, hue='metric', style='metric', markers=True, dashes=False, palette='muted')
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

# Customizing the plot
# plt.title('Bar Chart with Three Groups, Each Having Two Bars')
plt.xlabel('Percentage of challenge edges')
plt.ylabel(r'$P_{TP}$ or $P_{TN}$')
plt.xticks([0.3, 0.4, 0.5, 0.6, 0.7])
# plt.legend()
plt.legend([], [], frameon=False)
plt.savefig(f'./figures/mix_verification_lastfm.pdf', dpi=300, bbox_inches='tight')
labels = plt.gca().get_legend_handles_labels()
print('labels', labels)

# Show plot
plt.show()

fig_legend, ax_legend = plt.subplots(figsize=(10, 1))
ax_legend.axis('off')
legend_names = [r'$P_{TP}$', r'$P_{TN}$']
ax_legend.legend(labels[0][::-1], legend_names, loc='center', ncol=2)
fig_legend.savefig('./figures/mix_verification_legend.pdf', dpi=300, bbox_inches='tight')