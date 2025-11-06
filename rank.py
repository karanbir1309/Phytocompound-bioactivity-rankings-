#!/usr/bin/env python3
"""
Heterogeneous GNN Pipeline: RMSD=0 Docking + Average Binding + ADMET
Includes output of both average and best binding affinity with best target
Author: AI Research Assistant
Date: 2025-10-28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Step 1: ADMET scoring excluding LD50 oral toxicity
def calculate_admet_score(row):
    def safe_float(value):
        try:
            if isinstance(value, str):
                cleaned = value.strip("[]'\" ")
                if cleaned in ('-', ''):
                    return None
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return None

    score = 0
    max_score = 0

    if pd.notna(row['hia']):
        score += 1.5 if row['hia'] > 0.7 else (0.75 if row['hia'] > 0.5 else 0)
        max_score += 1.5

    if pd.notna(row['caco2']):
        score += 1.5 if row['caco2'] > -5 else (0.75 if row['caco2'] > -6 else 0)
        max_score += 1.5

    if pd.notna(row['BBB']):
        score += 1.0 if row['BBB'] == 'Low' else (0.5 if row['BBB'] == 'Medium' else 0)
        max_score += 1.0

    if pd.notna(row['PPB']):
        score += 1.0 if 60 <= row['PPB'] <= 95 else 0
        max_score += 1.0

    for key, weight in [('pgp_sub', 1.0), ('CYP3A4_sub', 0.5), ('CYP3A4_inh', 1.0)]:
        val = row.get(key)
        if isinstance(val, str):
            score += weight if val.lower() == 'no' else 0
            max_score += weight

    if pd.notna(row['t0.5']):
        score += 1.0 if 5 <= row['t0.5'] <= 15 else (0.5 if 2 <= row['t0.5'] <= 20 else 0)
        max_score += 1.0

    if pd.notna(row['cl-plasma']):
        score += 1.0 if 10 <= row['cl-plasma'] <= 50 else 0
        max_score += 1.0

    for tox_col in ['hERG', 'DILI', 'AMES']:
        val = row.get(tox_col)
        if val is not None:
            if tox_col == 'hERG':
                score += 3.0 if val == 'Low' else (1.5 if val == 'Medium' else 0)
                max_score += 3.0
            else:
                score += 3.0 if val == 'Negative' else 0
                max_score += 3.0

    if pd.notna(row['logP']):
        score += 1.0 if 0 <= row['logP'] <= 5 else (0.5 if -1 <= row['logP'] <= 6 else 0)
        max_score += 1.0

    if pd.notna(row['TPSA']):
        score += 1.0 if 20 <= row['TPSA'] <= 140 else 0
        max_score += 1.0

    return (score / max_score * 100) if max_score > 0 else 0

# Load ADMET
admet = pd.read_csv('phytocompounds_ADMET_clean.csv')
admet['ADMET_Score'] = admet.apply(calculate_admet_score, axis=1)
median_admet = admet['ADMET_Score'].median()

# Load main datasets
phytocompounds = pd.read_csv('phytocompounds.csv')
targets = pd.read_csv('targets.csv')
bioactivity_raw = pd.read_csv('bioactivity_merged.csv')

# Filter bioactivity for RMSD=0
bioactivity = bioactivity_raw[(bioactivity_raw['rmsd/ub'] == 0) & (bioactivity_raw[' rmsd/lb'] == 0)].copy()

bioactivity['Ligand_parsed'] = bioactivity['Ligand'].str.split('_', n=1)
bioactivity['pdb_id'] = bioactivity['Ligand_parsed'].apply(lambda x: x[0] if len(x)>0 else None)
bioactivity['compound_part'] = bioactivity['Ligand_parsed'].apply(lambda x: x[1] if len(x)>1 else None)
bioactivity['compound_name'] = bioactivity['compound_part'].str.split('_uff_E=').str[0]

pdb_to_target = targets.set_index('pdb_id')['target_name'].to_dict()
bioactivity['target_name'] = bioactivity['pdb_id'].map(pdb_to_target)

name_map = {'beta-sitosterol':'Beta-sitosterol', 'diosgenin':'Diosgenin', 'dioscin':'Dioscin'}
bioactivity['compound_name_mapped'] = bioactivity['compound_name'].apply(lambda x: name_map.get(x, x))
admet_dict = admet.set_index('compound_name')['ADMET_Score'].to_dict()
bioactivity['ADMET_Score'] = bioactivity['compound_name_mapped'].map(admet_dict).fillna(median_admet)

compound_list = sorted(bioactivity['compound_name'].unique())
compound_features = {}
for compound in compound_list:
    normalized = compound.lower().strip()
    row = admet[admet['compound_name'].str.lower().str.strip() == normalized]
    if not row.empty:
        row = row.iloc[0]
        compound_features[compound] = [
            row['caco2'], row['hia'], row['PPB'],
            row['logP'], row['logS'], row['TPSA'],
            row['ADMET_Score']/100.0
        ]
    else:
        compound_features[compound] = [
            admet['caco2'].mean(), admet['hia'].mean(), admet['PPB'].mean(),
            admet['logP'].mean(), admet['logS'].mean(), admet['TPSA'].mean(),
            median_admet/100.0
        ]

protein_le = LabelEncoder()
mode_le = LabelEncoder()
targets['protein_class_enc'] = protein_le.fit_transform(targets['protein_class'])
targets['mode_action_enc'] = mode_le.fit_transform(targets['Mode_of_action'])
target_features = {row['target_name']: [row['protein_class_enc'], row['mode_action_enc']] for _, row in targets.iterrows()}

compound_to_idx = {c:i for i, c in enumerate(compound_list)}
target_to_idx = {t:i for i, t in enumerate(target_features.keys())}

compound_feat_t = torch.FloatTensor([compound_features[c] for c in compound_list])
target_feat_t = torch.FloatTensor([target_features[t] for t in target_to_idx])

compound_feat_t = torch.FloatTensor(StandardScaler().fit_transform(compound_feat_t))
target_feat_t = torch.FloatTensor(StandardScaler().fit_transform(target_feat_t))

edges_c2t = []
attr_c2t = []

for _, row in bioactivity.iterrows():
    c = row['compound_name']
    t = row['target_name']
    if c in compound_to_idx and t in target_to_idx:
        edges_c2t.append([compound_to_idx[c], target_to_idx[t]])
        attr_c2t.append([abs(row['Binding Affinity']) / 15.0])

edge_index_c2t = torch.LongTensor(edges_c2t).t().contiguous()
edge_attr_c2t = torch.FloatTensor(attr_c2t)
edge_index_t2c = edge_index_c2t.flip([0])
edge_attr_t2c = edge_attr_c2t

data = HeteroData()
data['compound'].x = compound_feat_t
data['target'].x = target_feat_t

data['compound', 'binds_to', 'target'].edge_index = edge_index_c2t
data['compound', 'binds_to', 'target'].edge_attr = edge_attr_c2t
data['target', 'bound_by', 'compound'].edge_index = edge_index_t2c
data['target', 'bound_by', 'compound'].edge_attr = edge_attr_t2c

class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, num_layers=3):
        super().__init__()
        self.comp_lin = Linear(compound_feat_t.shape[1], hidden_channels)
        self.tgt_lin = Linear(target_feat_t.shape[1], hidden_channels)
        self.convs = nn.ModuleList([
            HeteroConv({
                ('compound','binds_to','target'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                ('target','bound_by','compound'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            }, aggr='mean') for _ in range(num_layers)
        ])
        self.comp_out = nn.Sequential(
            Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_channels//2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'compound': F.relu(self.comp_lin(x_dict['compound'])),
            'target': F.relu(self.tgt_lin(x_dict['target']))
        }
        for conv in self.convs:
            x_new = conv(x_dict, edge_index_dict)
            for k in x_dict:
                if k in x_new:
                    x_dict[k] = F.relu(x_new[k]) + x_dict[k]
        return self.comp_out(x_dict['compound'])

model = HeteroGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.MSELoss()

compound_scores = {}
for compound in compound_list:
    compound_data = bioactivity[bioactivity['compound_name'] == compound]
    compound_admet = compound_data['ADMET_Score'].iloc[0]

    if len(compound_data) > 0:
        avg_binding = compound_data['Binding Affinity'].mean()
        n_strong = np.sum(compound_data['Binding Affinity'] <= -10.0)

        binding_score = (abs(avg_binding) - 6.0) / 8.0
        binding_score = max(0, min(1, binding_score))

        admet_score = compound_admet / 100.0
        multitarget_score = n_strong / 13.0

        final_score = 0.4 * binding_score + 0.4 * admet_score + 0.2 * multitarget_score
    else:
        final_score = 0.0

    compound_scores[compound] = final_score

labels = torch.FloatTensor([compound_scores[c] for c in compound_list]).unsqueeze(1)

x_dict = {'compound': data['compound'].x, 'target': data['target'].x}
edge_index_dict = {
    ('compound', 'binds_to', 'target'): data['compound', 'binds_to', 'target'].edge_index,
    ('target', 'bound_by', 'compound'): data['target', 'bound_by', 'compound'].edge_index,
}

num_epochs = 200
train_losses = []

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(x_dict, edge_index_dict)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    predictions = model(x_dict, edge_index_dict).squeeze()

ranking_data = []
for i, compound in enumerate(compound_list):
    compound_data = bioactivity[bioactivity['compound_name']==compound]
    best_binding_val = compound_data['Binding Affinity'].min()
    best_target_name = compound_data.loc[compound_data['Binding Affinity'].idxmin()]['target_name']
    rank_entry = {
        'Rank': 0,
        'Compound': compound,
        'GNN_Score': predictions[i].item(),
        'True_Score': labels[i].item(),
        'Best_Binding': best_binding_val,
        'Best_Target': best_target_name,
        'Avg_Binding': compound_data['Binding Affinity'].mean(),
        'ADMET_Score': compound_data['ADMET_Score'].iloc[0],
        'N_Strong_Binders': np.sum(compound_data['Binding Affinity'] <= -10.0)
    }
    ranking_data.append(rank_entry)

rank_df = pd.DataFrame(ranking_data).sort_values('GNN_Score', ascending=False).reset_index(drop=True)
rank_df['Rank'] = range(1, len(rank_df) + 1)

print(rank_df[['Rank', 'Compound', 'GNN_Score', 'Best_Binding', 'Best_Target', 'Avg_Binding', 'ADMET_Score', 'N_Strong_Binders']].to_string(index=False))

print("\nModel Performance:")
print(f"MSE: {mean_squared_error(rank_df['True_Score'], rank_df['GNN_Score']):.6f}")
print(f"MAE: {mean_absolute_error(rank_df['True_Score'], rank_df['GNN_Score']):.6f}")
print(f"R²: {r2_score(rank_df['True_Score'], rank_df['GNN_Score']):.4f}")

sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs+1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss')
plt.tight_layout()
plt.savefig('training_loss.png')
plt.close()

plt.figure(figsize=(12,8))
colors = plt.cm.viridis(np.linspace(0, 1, len(rank_df)))
bars = plt.barh(rank_df['Compound'], rank_df['GNN_Score'], color=colors)
for bar, score in zip(bars, rank_df['GNN_Score']):
    plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.4f}', va='center')
plt.xlabel('GNN Score')
plt.title('Final Compound Rankings')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('final_rankings.png')
plt.close()

rank_df.to_csv('final_gnn_rankings_with_admet.csv', index=False)
print("Pipeline complete. Outputs saved:")
print(" - final_gnn_rankings_with_admet.csv")
print(" - training_loss.png")
print(" - final_rankings.png")