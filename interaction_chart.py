import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
bioactivity = pd.read_csv('bioactivity_merged.csv')
targets = pd.read_csv('targets.csv')
rankings = pd.read_csv('final_gnn_rankings_with_admet.csv')  # Load your rankings

# Parse compound and target names
bioactivity['compound_name'] = bioactivity['Ligand'].str.split('_', n=1).str[1].str.split('_uff_E=').str[0]
bioactivity['pdb_id'] = bioactivity['Ligand'].str.split('_', n=1).str[0]
pdb_to_target = targets.set_index('pdb_id')['target_id'].to_dict()
bioactivity['target_id'] = bioactivity['pdb_id'].map(pdb_to_target)

# Select top 5 compounds from rankings
top_5_compounds = rankings.head(5)['Compound'].tolist()

# Create pivot table for top 5 compounds
pivot = bioactivity[bioactivity['compound_name'].isin(top_5_compounds)].pivot_table(
    index='compound_name', columns='target_id', values='Binding Affinity', aggfunc='mean'
)

# Fill missing values and convert to positive scale
pivot = pivot.fillna(0).abs()

# Normalize to 0-1 scale
pivot_norm = pivot.div(pivot.max(axis=1), axis=0)

# Radar chart setup
labels = pivot_norm.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Plot each compound
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for idx, compound in enumerate(pivot_norm.index):
    values = pivot_norm.loc[compound].tolist()
    values += values[:1]
    ax.plot(angles, values, label=compound, linewidth=2, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
plt.title('Top 5 Compounds: Target Binding Profile (Radar Chart)', fontsize=16, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout()
plt.savefig('radar_chart_top5_compounds.png', dpi=300)
plt.show()
