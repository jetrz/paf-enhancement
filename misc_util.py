import torch, pickle, os, dgl
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dgl

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


# directory_path = f'static/graphs/ghost2-1/'
# rename_files(directory_path, "ghost2-1_", "")

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
    
    if mode == 'ghost2':
        edge_attrs.append('is_real_edge')
        node_attrs.append('is_real_node')
    
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

# mode = 'ghost2-1'

# ref = {}
# for i in [1,3,5,9,12,18]:
#     ref[i] = [i for i in range(15)]
# for i in [11,16,17,19,20]:
#     ref[i] = [i for i in range(5)]

# for chr in [1,3,5,9,12,18,11,16,17,19,20]:
#     for i in ref[chr]:
#         g_name = f'chr{chr}_M_{i}'
#         print(f"Converting {g_name}...")
#         path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/{mode}/'
#         pyg_g = torch.load(path + g_name + '.pt')
#         dgl_g = pyg_to_dgl(pyg_g, True, mode)
#         dgl.save_graphs(path + g_name + '.dgl', [dgl_g])

# for g_name in ['arab', 'chicken']:
#     print(f"Converting {g_name}...")
#     path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/{mode}/'
#     pyg_g = torch.load(path + g_name + '.pt')
#     dgl_g = pyg_to_dgl(pyg_g, False, mode)
#     dgl.save_graphs(path + g_name + '.dgl', [dgl_g])



        
