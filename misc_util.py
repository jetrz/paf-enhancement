import torch, pickle, os, dgl
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dgl
from tqdm import tqdm

def compare():
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

def rename_files(directory, old, new):
    # List all files in the given directory
    for filename in os.listdir(directory):
        # Check if the filename ends with '_ghost.pt'
        new_filename = filename.replace(old, new)
        # Create full path to the old and new files
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


# directory_path = '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/no_inner_edges/'
# rename_files(directory_path, "_no_inner_edges", "")

def edge_exists(edge_index, src_id, dst_id):
    # Convert edge_index to a 2D numpy array for easier processing
    edge_pairs = edge_index.t().numpy()
    
    # Check if the (src_id, dst_id) pair exists in the array of edge pairs
    return any((edge[0] == src_id and edge[1] == dst_id) for edge in edge_pairs)

def to_tensor(x):
    if isinstance(x, list):
        return torch.tensor(x)
    return x

def pyg_to_dgl(g, train, mode):
    if train:
        edge_attrs=['prefix_length', 'overlap_length', 'overlap_similarity', 'y', 
            'gt_17c', 'gt_soft', 'gt_bin', 'decision_edge_gt_only_pos', 'decision_edge_gt_only_neg', 'E_ID']
        node_attrs=['read_length', 'read_strand', 'read_start', 
            'read_end', 'read_chr', 'decision_node', 'N_ID']
    else:
        edge_attrs=['prefix_length', 'overlap_length', 'overlap_similarity', 'E_ID']
        node_attrs=['read_length', 'N_ID']

    if mode.startswith('ghost'):
        edge_attrs.append('edge_hop')
        node_attrs.append('node_hop')
    
    print("PyG graph:", g)
    u, v = g.edge_index
    dgl_g = dgl.graph((u, v))

    # Adding node features
    for attr in node_attrs:
        dgl_g.ndata[attr] = to_tensor(g[attr])

    # Adding edge features
    for attr in edge_attrs:
        dgl_g.edata[attr] = to_tensor(g[attr])

    print("DGL graph:", dgl_g)
    return dgl_g

def pyg_to_nx(g):
    num_nodes, num_edges = g.N_ID.shape[0], g.E_ID.shape[0]

    node_attrs, edge_attrs = [], []
    for attr_name in g.keys():
        attr = getattr(g, attr_name)
        
        if attr.shape[0] == num_nodes:
            node_attrs.append(attr_name)
        elif attr.shape[0] == num_edges:
            edge_attrs.append(attr_name)


    nx_g = to_networkx(g, node_attrs=node_attrs, edge_attrs=edge_attrs)
    return nx_g

def analyse_walks(name):
    path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl"
    with open(path, 'rb') as f:
        walks = pickle.load(f)
        print(walks)
        print(len(walks))

# analyse_walks('chm13')

# for i in tqdm(range(15), ncols=120):
#     path = '../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/temp/'
#     g = torch.load(path+f'chr18_M_{i}.pt')
#     nx_g = pyg_to_nx(g)

#     with open(path+f"nx_chr18_M_{i}.pkl", "wb") as p:
#         pickle.dump(nx_g, p)