# Import packages
import pandas as pd
import numpy as np
import networkx as nx
import gc
from torch_geometric.utils import from_networkx
from torch import save
import argparse
from memory_profiler import profile

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process integers for train-test split.')
parser.add_argument('--min_preceding_timesteps_for_train', type=int, default=1, help='Minimum preceding timesteps for training')
parser.add_argument('--n_train_timesteps', type=int, default=3, help='Number of training timesteps')
args = parser.parse_args()

# Parameters
min_preceding_timesteps_for_train = args.min_preceding_timesteps_for_train
n_train_timesteps = args.n_train_timesteps

print("Loading data...")

# Load data
df = pd.read_csv('data/subnational/gid_timem_best.csv', engine='pyarrow').dropna(subset='best')

# Filter for gids that have more than 3 percent of timestamps
freq = (df.value_counts('gid') / df.value_counts('gid').max())
freq = freq.loc[freq == 1]
none_missing = freq.index.tolist()
df = df.query('gid.isin(@none_missing)')

df['timem'] = pd.to_datetime(df.timem)
df['log_best'] = np.log1p(df.best); df.drop(columns='best', inplace=True)
df.sort_values(by='timem', inplace=True)
newcols = pd.concat([df.groupby('gid').log_best.shift(-h).rename(f'y_{h}') for h in range(1, 7)], axis=1)
df = pd.concat([df, newcols], axis=1)
# df = df.loc[(df.timem.dt.year == 2019) & (df.timem.dt.month < 7)]  # for a small sample to test code
df.reset_index(drop=True, inplace=True)

time_dict = {key: val for key, val in zip(df.timem.drop_duplicates(), df.timem.drop_duplicates().dt.strftime('%Y_%m'))}
df['time_str'] = df.timem.map(time_dict)
df['gidtime_str'] = df.gid.astype(str) + df.time_str

# Add train and test attributes to DataFrame
df['preceding_timesteps'] = df.groupby('gid').cumcount()
df['train'] = (df['preceding_timesteps'] >= min_preceding_timesteps_for_train).astype(int)
df['test'] = (df['preceding_timesteps'] < min_preceding_timesteps_for_train).astype(int)

print("Processing node data...")

# Node data
node_list = df[['log_best', 'gidtime_str', 'train', 'test'] + newcols.columns.tolist()].set_index('gidtime_str'). \
    apply(lambda row: (row.name, {'log_best': row['log_best'], 'y': [row[f'y_{i}'] for i in range(1, 7)], 'train': row['train'], 'test': row['test']}), axis=1). \
    tolist()

none_missing = pd.Series(none_missing).astype(str)

# Edges between nodes (non-temporal edges)
coordinate_dict = pd.read_pickle('data/subnational/coordinates.pkl')
coordinate_dict = {key: value for (val, key), value in coordinate_dict.items() if val == 1}

tims = df.time_str.unique()
coordinate_dict_timestr = {str(key) + tim: [str(val) + tim for val in value] for key, value in coordinate_dict.items() for tim in tims}
spatial_edges = [(u, v) for u, values in coordinate_dict_timestr.items() for v in values]

del coordinate_dict_timestr; del coordinate_dict; gc.collect()

# Temporal edges
temporal_edges = []
for i in range(len(tims) - 1):
    current_edges = list(zip(none_missing + tims[i], none_missing + tims[i + 1]))
    temporal_edges.extend(current_edges)

print("Creating graph...")

def add_edges_in_batches(G, edges, batch_size=1000):
    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        G.add_edges_from(batch)
        print(f"Added batch {i // batch_size + 1}")

# Add nodes, spatial edges, and temporal edges to graph
g = nx.Graph()
g.add_nodes_from(node_list)
add_edges_in_batches(g, spatial_edges)
g = g.to_directed()
add_edges_in_batches(g, temporal_edges)

# Remove nodes with no data from graph
g_nodes = pd.Series(g.nodes())
empty_nodes = g_nodes.loc[g_nodes == {}].index.tolist()
g.remove_nodes_from(empty_nodes)

print("Converting to PyTorch Geometric data object...")

# Convert to a torch_geometric.data.Data instance and output via torch.save
pyg = from_networkx(g)
save(pyg, 'output/ucdp_graph.pt')

print("Process completed successfully.")