import re
from copy import deepcopy
from collections import defaultdict
from datetime import datetime

from Bio.Seq import Seq
import torch
import edlib
from tqdm import tqdm

from torch_geometric.data import Data

def only_from_gfa(gfa_path, get_similarities=False, is_hifiasm=True):
    # Loop through GFA. Assume first N lines start with "S"
    time_start = datetime.now()
    print("Reading GFA...")
    with open(gfa_path) as f:
        rows = f.readlines()

    n_rows = len(rows)    
    N_ID, E_ID = 0, 0

    read_to_node, node_to_read = {}, {}
    read_to_node2 = {}
    read_lengths, read_seqs = {}, {}  # Obtained from the GFA
    edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}
    edge_index = [[],[]]

    r_ind = 0
    while r_ind < n_rows:
        row = rows[r_ind].strip().split()
        tag = row.pop(0)

        if tag == 'S':
            if len(row) == 4: # Hifiasm
                s_id, seq, length, count = row
            elif len(row) == 3: # Raven
                s_id, seq, length = row
            else:
                raise Exception("Unknown GFA format!")

            real_id, virt_id = N_ID, N_ID + 1 # 1+, 1-
            N_ID += 2
            read_to_node[s_id] = (real_id, virt_id)
            node_to_read[real_id] = s_id
            node_to_read[virt_id] = s_id

            seq = Seq(seq)
            read_seqs[real_id] = str(seq)
            read_seqs[virt_id] = str(seq.reverse_complement())

            length = int(length[5:])
            read_lengths[real_id] = length
            read_lengths[virt_id] = length

            if s_id.startswith('utg'):
                # The issue here is that in some cases, one unitig can consist of more than one read
                # So this is the adapted version of the code that supports that
                # The only things of importance here are read_to_node2 dict (not overly used)
                # And id variable which I use for obtaining positions during training (for the labels)
                # I don't use it for anything else, which is good
                s_ids = []
                while rows[r_ind+1][0] == 'A':
                    r_ind += 1
                    row = rows[r_ind].strip().split()
                    read_orientation, utg_to_read = row[3], row[4]
                    s_ids.append((utg_to_read, read_orientation))
                    read_to_node2[utg_to_read] = (real_id, virt_id)

                s_id = s_ids
                node_to_read[real_id] = s_id
                node_to_read[virt_id] = s_id
        elif tag == 'L':
            if len(row) == 5: # raven, normal GFA 1 standard
                s_id1, orient1, s_id2, orient2, cigar = row
            elif len(row) == 6: # Hifiasm GFA
                s_id1, orient1, s_id2, orient2, cigar, _ = row
                s_id1 = re.findall(r'(.*):\d-\d*', s_id1)[0]
                s_id2 = re.findall(r'(.*):\d-\d*', s_id2)[0]
            elif len(row) == 7: # Hifiasm GFA newer
                s_id1, orient1, s_id2, orient2, cigar, _, _ = row
            else:
                raise Exception("Unknown GFA format!")
            
            if orient1 == '+' and orient2 == '+':
                src_real = read_to_node[s_id1][0]
                dst_real = read_to_node[s_id2][0]
                src_virt = read_to_node[s_id2][1]
                dst_virt = read_to_node[s_id1][1]
            elif orient1 == '+' and orient2 == '-':
                src_real = read_to_node[s_id1][0]
                dst_real = read_to_node[s_id2][1]
                src_virt = read_to_node[s_id2][0]
                dst_virt = read_to_node[s_id1][1]
            elif orient1 == '-' and orient2 == '+':
                src_real = read_to_node[s_id1][1]
                dst_real = read_to_node[s_id2][0]
                src_virt = read_to_node[s_id2][1]
                dst_virt = read_to_node[s_id1][0]
            elif orient1 == '-' and orient2 == '-':
                src_real = read_to_node[s_id1][1]
                dst_real = read_to_node[s_id2][1]
                src_virt = read_to_node[s_id2][0]
                dst_virt = read_to_node[s_id1][0]  
            else:
                raise Exception("Unknown GFA format!")
            
            if is_hifiasm: # Don't need to manually add reverse complement edge
                edge_index[0].append(src_real)
                edge_index[1].append(dst_real)
                edge_ids[(src_real, dst_real)] = E_ID
                E_ID += 1
            else:
                edge_index[0].extend([src_real, src_virt])
                edge_index[1].extend([dst_real, dst_virt])
                edge_ids[(src_real, dst_real)] = E_ID
                edge_ids[(src_virt, dst_virt)] = E_ID + 1
                E_ID += 2

            # -----------------------------------------------------------------------------------
            # This enforces similarity between the edge and its "virtual pair"
            # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
            # When parsing CSV that was not necessarily so:
            # Sometimes reads would be slightly differently aligned from their RC pairs
            # Thus resulting in different overlap lengths
            # -----------------------------------------------------------------------------------

            try:
                ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
            except ValueError:
                print('Cannot convert CIGAR string into overlap length!')
                raise ValueError
            
            overlap_lengths[(src_real, dst_real)] = ol_length
            overlap_lengths[(src_virt, dst_virt)] = ol_length

            prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
            prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length

        r_ind += 1

    elapsed = (datetime.now() - time_start).seconds
    print(f"Run Time: {elapsed}s, Creating graph...")

    if get_similarities:
        print("Calculating similarities...")
        overlap_similarities = calculate_similarities(edge_ids, read_seqs, overlap_lengths)

    g = Data(N_ID=torch.tensor([i for i in range(N_ID)]), E_ID=torch.tensor([i for i in range(E_ID)]), edge_index=torch.tensor(edge_index))
    aux = { 'read_lengths_dict' : read_lengths, 'prefix_lengths_dict' : prefix_lengths, 'overlap_lengths_dict' : overlap_lengths }

    node_attrs, edge_attrs = ['N_ID', 'read_length'], ['E_ID', 'prefix_length', 'overlap_length']
    # Only convert to list right before creating graph data
    read_lengths_list = [read_lengths[i] for i in range(N_ID)]
    prefix_lengths_list, overlap_lengths_list = [0]*E_ID, [0]*E_ID
    for k, e_id in edge_ids.items():
        prefix_lengths_list[e_id] = prefix_lengths[k]
        overlap_lengths_list[e_id] = overlap_lengths[k]
    g['read_length'] = torch.tensor(read_lengths_list)
    g['prefix_length'] = torch.tensor(prefix_lengths_list)
    g['overlap_length'] = torch.tensor(overlap_lengths_list)

    if get_similarities:
        overlap_similarities_list = [0]*E_ID
        for k, e_id in edge_ids.items():
            overlap_similarities_list[e_id] = overlap_similarities[k]
        g['overlap_similarity'] = torch.tensor(overlap_similarities_list)
        aux['overlap_similarities_dict'] = overlap_similarities
        edge_attrs.append('overlap_similarity')

    # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
    if len(read_to_node2) != 0:
        read_to_node = read_to_node2

    aux = {
        'read_to_node' : read_to_node,
        'read_seqs' : read_seqs,
        'node_to_read' : node_to_read,
        'node_attrs' : node_attrs,
        'edge_attrs' : edge_attrs,
        'successor_dict' : graph_to_successor_dict(g)
    }

    return g, aux

def calculate_similarities(edge_ids, read_seqs, overlap_lengths):
    # Make sure that read_seqs is a dict of string, not Bio.Seq objects!
    overlap_similarities = {}
    for src, dst in tqdm(edge_ids.keys(), ncols=120):
        ol_length = overlap_lengths[(src, dst)]
        read_src = read_seqs[src]
        read_dst = read_seqs[dst]
        edit_distance = edlib.align(read_src[-ol_length:], read_dst[:ol_length])['editDistance']
        overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length
    return overlap_similarities

def reverse_graph(g):
    """
    Reverses a PyG graph by swapping rows in edge_index.
    """
    g_copy = deepcopy(g.detach())
    rev_e_index = g_copy.edge_index[[1,0], :]
    g_copy.edge_index = rev_e_index
    return g_copy

def graph_to_successor_dict(g):
    # Ensure the edge_index is in COO format and directed
    edge_index = g.edge_index
    successors_dict = defaultdict(list)

    # edge_index[0] contains source nodes, edge_index[1] contains target nodes
    for src, tgt in zip(edge_index[0], edge_index[1]):
        successors_dict[src.item()].append(tgt.item())

    return successors_dict