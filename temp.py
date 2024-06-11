import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

if __name__ == "__main__":
    edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3],
    [1, 0, 2, 1, 3, 2],
    ])
    data = Data(edge_index=edge_index, num_nodes=4)
    res = to_networkx(data)
    print(res)

