"""Define the retained GCN regression architecture and its training wrapper.

Inputs are molecular graphs with 37 node features and graph connectivity.
The scalar output predicts y = -log10(concentration in nM). Six bond features
are generated elsewhere and passed through the wrapper, but this GCN does not
use them. The network combines GCN convolutions, Graph Multiset Transformer
pooling and fully connected layers; it requires the pinned Torch/PyG runtime.

Import with 'from MoleculeACE.models import GCN'. The campaign entry point
is run_benchmark.py; this module does not start training or write outputs
when imported. See docs/protocol.md and docs/upstream_changes.md.

Author: Derek van Tilborg -- TU/e -- 19-05-2022.
Architecture references: Kipf & Welling (2017), Semi-Supervised Classification
with Graph Convolutional Networks; Baek et al. (2021), Accurate Learning of
Graph Representations with Graph Multiset Pooling.
"""

from MoleculeACE.benchmark.const import RANDOM_SEED
from MoleculeACE.models.utils import GNN
from torch_geometric.nn import GCNConv, GraphMultisetTransformer
from torch.nn import Linear, Dropout
import torch.nn.functional as F
import torch


class GCN(GNN):
    """Configure the GCN, Adam optimiser and MSE loss for one regression task."""
    def __init__(self, node_feat_in: int = 37, node_hidden: int = 128, transformer_hidden: int = 128,
                 n_conv_layers: int = 4, n_fc_layers: int = 4, fc_hidden: int = 128, lr: float = 0.0005,
                 epochs: int = 300, dropout: float = 0.2, *args, **kwargs):
        """Construct the model on the available CUDA device, otherwise CPU.

        Parameters
        ----------
        node_feat_in, node_hidden, transformer_hidden, fc_hidden : int
            Input and hidden feature widths; the retained featuriser produces 37 inputs.
        n_conv_layers, n_fc_layers : int
            Numbers of graph-convolution and fully connected hidden layers.
        lr : float
            Adam learning rate.
        epochs : int
            Maximum epochs used when training receives no explicit limit.
        dropout : float
            Dropout probability after each fully connected hidden layer.
        *args, **kwargs
            Retained upstream compatibility arguments; additional values are not used.

        Notes
        -----
        GCNmodel initialises Torch with seed 42. The campaign passes the published
        configuration after removing its epochs entry, preserving the default
        300-epoch benchmark ceiling. Construction alone does not train the model.
        """
        super().__init__()

        self.model = GCNmodel(node_feat_in=node_feat_in, node_hidden=node_hidden, n_conv_layers=n_conv_layers,
                              transformer_hidden=transformer_hidden, fc_hidden=fc_hidden, n_fc_layers=n_fc_layers,
                              dropout=dropout)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.name = 'GCN'

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    def __repr__(self):
        return f"{self.model}"


class GCNmodel(torch.nn.Module):
    """Apply graph convolution, eight-head multiset pooling and scalar regression."""
    def __init__(self, node_feat_in: int = 37, n_conv_layers: int = 4, n_fc_layers: int = 1, node_hidden: int = 128,
                 fc_hidden: int = 128, transformer_hidden: int = 128, dropout: float = 0.2, seed: int = RANDOM_SEED,
                 *args, **kwargs):

        """Create the retained layers and initialise Torch's random state.

        Parameters
        ----------
        node_feat_in, node_hidden, fc_hidden, transformer_hidden : int
            Node-input and hidden feature widths.
        n_conv_layers, n_fc_layers : int
            Numbers of convolutional and fully connected hidden layers.
        dropout : float
            Dropout probability in the fully connected layers.
        seed : int
            Torch initialisation seed; the GCN wrapper uses the default 42.
        *args, **kwargs
            Retained compatibility arguments, not used to configure further layers.
        """
        super().__init__()
        torch.manual_seed(seed)

        # GCN layer(s)
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feat_in, node_hidden))
        for k in range(n_conv_layers-1):
            self.conv_layers.append(GCNConv(node_hidden, node_hidden))

        # Global pooling
        self.transformer = GraphMultisetTransformer(in_channels=node_hidden,
                                                    hidden_channels=transformer_hidden,
                                                    out_channels=fc_hidden,
                                                    num_heads=8)

        # fully connected layer(s)
        self.fc = torch.nn.ModuleList()
        for k in range(n_fc_layers):
            self.fc.append(Linear(fc_hidden, fc_hidden))

        self.dropout = Dropout(dropout)

        # Output layer
        self.out = Linear(fc_hidden, 1)

    def forward(self, x, edge_index, edge_attr, batch):

        """Predict one logarithmic response for each graph in a batch.

        Parameters
        ----------
        x : torch.Tensor, shape (n_nodes, 37)
            Node features, converted to floating point before convolution.
        edge_index : torch.Tensor, shape (2, n_directed_edges)
            Connectivity; each molecular bond is represented in both directions.
        edge_attr : torch.Tensor
            Accepted for the shared graph-model interface but not used by this GCN.
        batch : torch.Tensor, shape (n_nodes,)
            Graph index for each node, used by the pooling layer.

        Returns
        -------
        torch.Tensor, shape (n_graphs, 1)
            Predictions of y = -log10(concentration in nM), not pKi/pEC50 values.
        """
        h = F.relu(self.conv_layers[0](x.float(), edge_index))
        for k in range(len(self.conv_layers) - 1):
            h = F.relu(self.conv_layers[k+1](h, edge_index))

        # Global graph pooling with a transformer
        h = self.transformer(h, batch, edge_index)

        # Apply a fully connected layer.
        for k in range(len(self.fc)):
            h = F.relu(self.fc[k](h))
            h = self.dropout(h)

        out = self.out(h)

        return out






