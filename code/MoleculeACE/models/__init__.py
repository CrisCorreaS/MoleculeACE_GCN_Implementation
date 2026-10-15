"""Load the retained GCN lazily while keeping protocol helpers Torch-free.

'from MoleculeACE.models import GCN' loads the pinned Torch/PyG model runtime.
Importing models.training does not request the model. The enclosing package
still requires NumPy, pandas and PyYAML; import does not train or write files.
"""
def __getattr__(name):
    if name == 'GCN':
        from MoleculeACE.models.gcn import GCN
        return GCN
    raise AttributeError(f"module 'MoleculeACE.models' has no attribute {name!r}")
