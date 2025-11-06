import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
bioactivity = pd.read_csv('bioactivity_merged.csv')
targets = pd.read_csv('targets.csv')

# Parse compound and target names
bioactivity['compound_name'] = bioactivity['Ligand'].str.split('_', n=1).str[1].str.split('_uff_E=').str[0]
bioactivity['pdb_id'] = bioactivity['Ligand'].str.split('_', n=1).str[0]
pdb_to_target = targets.set_index('pdb_id')['target_id'].to_dict()
bioactivity['target_id'] = bioactivity['pdb_id'].map(pdb_to_target)

# Create pivot table: rows=compounds, cols=targets, values=binding affinity
pivot = bioactivity.pivot_table(index='compound_name', columns='target_id', values='Binding Affinity', aggfunc='mean')

plt.figure(figsize=(14, 8))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='coolwarm_r', linewidths=0.5)
plt.title('Compound-Target Binding Affinity Heatmap (kcal/mol)', fontsize=16)
plt.xlabel('Targets', fontsize=12)
plt.ylabel('Compounds', fontsize=12)
plt.tight_layout()
plt.savefig('heatmap_compound_target.png', dpi=300)
plt.show()