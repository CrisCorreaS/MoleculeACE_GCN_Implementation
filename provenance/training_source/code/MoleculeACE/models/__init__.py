# Upstream imported all 12 models here. Only GCN is kept, exported lazily (PEP 562) so that the torch-free
# MoleculeACE.models.training can be imported without torch/torch_geometric installed.
def __getattr__(name):
    if name == 'GCN':
        from MoleculeACE.models.gcn import GCN
        return GCN
    raise AttributeError(f"module 'MoleculeACE.models' has no attribute {name!r}")
