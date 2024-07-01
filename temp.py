import torch, pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

if __name__ == "__main__":
    pyg = torch.load('static/graphs/chr18_M_14_ghost.pt')
    dgl_edge_index = torch.load('dgl_edge_index_temp.pt')
    dgl_ol_len = torch.load('dgl_ol_len_temp.pt')
    dgl_ol_sim = torch.load('dgl_ol_sim_temp.pt')
    dgl_prefix_len = torch.load('dgl_prefix_len_temp.pt')
    n_edges_dgl = len(dgl_edge_index[0]) # dgl graph will have less edges because pyg is enhanced

    print("Comparing Edge Index 0")
    check = torch.eq(pyg.edge_index[0][:n_edges_dgl], dgl_edge_index[0]).tolist()
    print("Len:", len(check), "True:", check.count(True), 'False:', check.count(False))

    print("Comparing Edge Index 1")
    check = torch.eq(pyg.edge_index[1][:n_edges_dgl], dgl_edge_index[1]).tolist()
    print("Len:", len(check), "True:", check.count(True), 'False:', check.count(False))
    index = check.index(False)
    print("pyg:", pyg.edge_index[1][index], "dgl:", dgl_edge_index[1][index])

    print("Comparing OL Len")
    one, _ = pyg.overlap_length[:n_edges_dgl].sort()
    two, _ = dgl_ol_len.sort()
    check = torch.eq(one.float(), two.float()).tolist()
    print("Len:", len(check), "True:", check.count(True), 'False:', check.count(False))

    print("Comparing OL Sim")
    one, _ = torch.sort(pyg.overlap_similarity[:n_edges_dgl])
    two, _ = torch.sort(dgl_ol_sim)
    check = torch.eq(one.float(), two.float()).tolist()
    print("Len:", len(check), "True:", check.count(True), 'False:', check.count(False))

    print("Comparing Prefix Length")
    one, _ = torch.sort(pyg.prefix_length[:n_edges_dgl])
    two, _ = torch.sort(dgl_prefix_len)
    check = torch.eq(one.float(), two.float()).tolist()
    print("Len:", len(check), "True:", check.count(True), 'False:', check.count(False))