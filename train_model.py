#%%
from torch import save, load
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

# Load data
data = load('output/ucdp_graph.pt')
data.log_best = data.log_best.view(data.log_best.shape[0],1)

# Ensure input nodes are of type torch.long
data.train = data.train.to(torch.long)
data.val = data.val.to(torch.long)
data.test = data.test.to(torch.long)

#%%
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """GraphSAGE with an arbitrary number of layers for regression tasks."""
    def __init__(self, dim_in, dim_h, dim_out, num_layers, edge_types):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.edge_types = edge_types
        self.num_edge_types = len(edge_types)
        
        # Define separate aggregators for each edge type and layer
        self.aggregators = torch.nn.ModuleList()
        for _ in range(num_layers):
            layer = torch.nn.ModuleList()
            for _edge_types in range(self.num_edge_types):
                layer.append(SAGEConv(dim_in if _ == 0 else dim_h * self.num_edge_types, dim_h))
            self.aggregators.append(layer)

        # MLP for final prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_h * self.num_edge_types, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, dim_out)
        )

    def forward(self, x, edge_index, edge_weight):
        for layer in self.aggregators:
            embeddings = []
            # Perform message passing for each edge type
            for i, edge_type in enumerate(self.edge_types):
                mask = (edge_weight == edge_type)
                filtered_edge_index = edge_index[:, mask]

                h = x

                h = layer[i](h, filtered_edge_index)
                h = torch.relu(h)
                h = F.dropout(h, p=0.5, training=self.training)
                embeddings.append(h)

            # Concatenate embeddings
            x = torch.cat(embeddings, dim=1)

        return x

    def predict(self, x):
        # Pass through MLP for final prediction
        out = self.mlp(x)
        return out

    def fit(self, data, epochs, train_loader, val_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                embeddings = self(batch.log_best, batch.edge_index, batch.weight)
                out = self.predict(embeddings)
                loss = criterion(out[batch.train], batch.y[batch.train])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

            self.eval()
            total_val_loss = 0
            for batch in val_loader:
                embeddings = self(batch.log_best, batch.edge_index, batch.weight)
                out = self.predict(embeddings)
                val_loss = criterion(out[batch.val], batch.y[batch.val])
                total_val_loss += val_loss.item()
            print(f'Validation Loss: {total_val_loss / len(val_loader)}')
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Ensure data is on the correct device
data = data.to(device)

# Model parameters
dim_in, dim_h, dim_out, num_layers = 1, 10, 6, 1
epochs = 10
edge_types = data.weight.unique()

# Data loaders
train_loader = NeighborLoader(data, batch_size=1024, input_nodes=data.train, num_neighbors=[-1] * num_layers)
val_loader = NeighborLoader(data, batch_size=data.val.sum().item(), input_nodes=data.test, num_neighbors=[-1] * num_layers)

# Initialize model, move to device, and fit
model = GraphSAGE(dim_in, dim_h, dim_out, num_layers, data.weight.unique().tolist()).to(device)
model.fit(data, epochs, train_loader, val_loader)
# %%
save(model, 'output/model.pt')
