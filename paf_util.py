from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import edlib
import torch
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool

READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ = None, None, None, None, None
CASES = { i:0 for i in range(3) }

# We have to do this because cannot pickle defaultdicts created by lambda
def create_list_dd():
    return defaultdict(list)

def preprocess_rows(rows): 
    res, dupchecker, utgchecker, ghost_utg_checker = [], set(), set(), { '+' : defaultdict(set), '-' : defaultdict(set) }
    duplicates, rejected = 0, 0
    print("Preprocessing paf...")
    for row in tqdm(rows, ncols=120):
        row_split = row.strip().split()
        id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row_split

        if (id1, id2) in dupchecker or (id2, id1) in dupchecker:
            duplicates += 1
            continue
        else:
            len1, start1, end1, len2, start2, end2 = int(len1), int(start1), int(end1), int(len2), int(start2), int(end2)
            src, dst = None, None
            if orientation == '+':
                if start1 == 0 and start2 == 0:
                    rejected += 1
                    continue
                elif end1 == len1 and end2 == len2:
                    rejected += 1
                    continue
                elif end1 == len1 and start2 == 0:
                    src, dst = (id1, '+'), (id2, '+')
                    src_rev, dst_rev = (id2, '-'), (id1, '-')
                elif start1 == 0 and end2 == len2:
                    src, dst = (id2, '+'), (id1, '+')
                    src_rev, dst_rev = (id1, '-'), (id2, '-')
                else:
                    rejected += 1
                    continue
            else:
                if start1 == 0 and end2 == len2:
                    rejected += 1
                    continue
                elif end1 == len1 and start2 == 0:
                    rejected += 1
                    continue
                elif end1 == len1 and end2 == len2:
                    src, dst = (id1, '+'), (id2, '-')
                    src_rev, dst_rev = (id2, '+'), (id1, '-')
                elif start1 == 0 and start2 == 0:
                    src, dst = (id1, '-'), (id2, '+')
                    src_rev, dst_rev = (id2, '-'), (id1, '+')
                else:
                    rejected += 1
                    continue

            src_id, dst_id = src[0], dst[0]

            # handling edge cases from unitigs
            if src_id in READ_TO_NODE and dst_id in READ_TO_NODE:
                nids1, nids2 = READ_TO_NODE[src_id], READ_TO_NODE[dst_id]

                # Overlaps between reads in the same unitig 
                if nids1 == nids2:
                    duplicates += 1
                    continue

                # Overlaps where one or both nodes are unitigs
                if src[1] == '+' and dst[1] == '+':
                    src_n_id, dst_n_id = READ_TO_NODE[src_id][0], READ_TO_NODE[dst_id][0]
                    src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][1], READ_TO_NODE[dst_rev[0]][1]
                elif src[1] == '+' and dst[1] == '-':
                    src_n_id, dst_n_id = READ_TO_NODE[src_id][0], READ_TO_NODE[dst_id][1]
                    src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][0], READ_TO_NODE[dst_rev[0]][1]
                elif src[1] == '-' and dst[1] == '+':
                    src_n_id, dst_n_id = READ_TO_NODE[src_id][1], READ_TO_NODE[dst_id][0]
                    src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][1], READ_TO_NODE[dst_rev[0]][0]

                src_reads, dst_reads = NODE_TO_READ[src_n_id], NODE_TO_READ[dst_n_id]
                if isinstance(src_reads, list) and len(src_reads) > 1:
                    if src_id != src_reads[0][0] and src_id != src_reads[-1][0]:
                        rejected += 1
                        continue

                    if src_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if src_id == src_reads[0][0]:
                        if src_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if src_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                if isinstance(dst_reads, list) and len(dst_reads) > 1:
                    if dst_id != dst_reads[0][0] and dst_id != dst_reads[-1][0]:
                        rejected += 1
                        continue

                    if dst_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if dst_id == dst_reads[0][0]:
                        if dst_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if dst_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                if (src_n_id, dst_n_id) in utgchecker or (src_rev_n_id, dst_rev_n_id) in utgchecker:
                    # if READ_TO_NODE[id1] != READ_TO_NODE[id2]: print("case caught. src_n_id:", src_n_id, "dst_n_id:", dst_n_id)
                    duplicates += 1
                    continue
                else:
                    utgchecker.add((src_n_id, dst_n_id))

            elif src_id in READ_TO_NODE:
                if src[1] == '+':
                    src_n_id = READ_TO_NODE[src_id][0]
                else:
                    src_n_id = READ_TO_NODE[src_id][1]

                src_reads = NODE_TO_READ[src_n_id]

                if isinstance(src_reads, list) and len(src_reads) > 1:
                    if src_id != src_reads[0][0] and src_id != src_reads[-1][0]:
                        rejected += 1
                        continue

                    if src_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if src_id == src_reads[0][0]:
                        if src_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if src_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                    if src_n_id in ghost_utg_checker[dst[1]][dst_id]:
                        duplicates += 1
                        continue
                    else:
                        ghost_utg_checker[dst[1]][dst_id].add(src_n_id)

            elif dst_id in READ_TO_NODE:
                if dst[1] == '+':
                    dst_n_id = READ_TO_NODE[dst_id][0]
                else:
                    dst_n_id = READ_TO_NODE[dst_id][1]

                dst_reads = NODE_TO_READ[dst_n_id]

                if isinstance(dst_reads, list) and len(dst_reads) > 1:
                    if dst_id != dst_reads[0][0] and dst_id != dst_reads[-1][0]:
                        rejected += 1
                        continue

                    if dst_id == id1:
                        c_start, c_end, c_len = start1, end1, len1
                    else:
                        c_start, c_end, c_len = start2, end2, len2

                    if dst_id == dst_reads[0][0]:
                        if dst_reads[0][1] == '+':
                            if c_start != 0:
                                rejected += 1
                                continue
                        else:
                            if c_end != c_len:
                                rejected += 1
                                continue
                    else:
                        if dst_reads[-1][1] == '+':
                            if c_end != c_len:
                                rejected += 1
                                continue
                        else:
                            if c_start != 0:
                                rejected += 1
                                continue

                    if dst_n_id in ghost_utg_checker[src[1]][src_id]:
                        duplicates += 1
                        continue
                    else:
                        ghost_utg_checker[src[1]][src_id].add(dst_n_id)

            dupchecker.add((id1, id2))
            res.append(row)

    print("Preprocessing done! Number of duplicates:", duplicates, "Number of rejected:", rejected)
    return res

# For multiprocessing
def parse_row(row):
    '''
    Returns
    'code' : 0 if rejected, 1 if both src and dst are in gfa, 2 if only either src or dst is in gfa
    'data' : None if code == 0, respective information otherwise
    '''
    data = None

    if not READ_SEQS or not READ_TO_NODE or not ANNOTATED_FASTA_DATA or not SUCCESSOR_DICT:
        raise ValueError("Global objects not set!")

    row = row.strip().split()

    ## What are these last 3 fields? ##
    id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row
    len1, start1, end1, len2, start2, end2 = int(len1), int(start1), int(end1), int(len2), int(start2), int(end2)

    src, dst = None, None
    if orientation == '+':
        if start1 == 0 and start2 == 0:
            return 0, data
        elif end1 == len1 and end2 == len2:
            return 0, data
        elif end1 == len1 and start2 == 0:
            src, dst = (id1, '+'), (id2, '+')
            src_rev, dst_rev = (id2, '-'), (id1, '-')
        elif start1 == 0 and end2 == len2:
            src, dst = (id2, '+'), (id1, '+')
            src_rev, dst_rev = (id1, '-'), (id2, '-')
        else:
            return 0, data
    else:
        if start1 == 0 and end2 == len2:
            return 0, data
        elif end1 == len1 and start2 == 0:
            return 0, data
        elif end1 == len1 and end2 == len2:
            src, dst = (id1, '+'), (id2, '-')
            src_rev, dst_rev = (id2, '+'), (id1, '-')
        elif start1 == 0 and start2 == 0:
            src, dst = (id1, '-'), (id2, '+')
            src_rev, dst_rev = (id2, '-'), (id1, '+')
        else:
            return 0, data
    
    src_id, dst_id = src[0], dst[0]
    
    if src_id not in READ_TO_NODE and dst_id not in READ_TO_NODE: return 0, data
        
    if src[1] == '+' and dst[1] == '+':
        src_seq = READ_SEQS[READ_TO_NODE[src_id][0]] if src_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[src_id][0]
        dst_seq = READ_SEQS[READ_TO_NODE[dst_id][0]] if dst_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[dst_id][0]
    elif src[1] == '+' and dst[1] == '-':
        src_seq = READ_SEQS[READ_TO_NODE[src_id][0]] if src_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[src_id][0]
        dst_seq = READ_SEQS[READ_TO_NODE[dst_id][1]] if dst_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[dst_id][1]
    elif src[1] == '-' and dst[1] == '+':
        src_seq = READ_SEQS[READ_TO_NODE[src_id][1]] if src_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[src_id][1]
        dst_seq = READ_SEQS[READ_TO_NODE[dst_id][0]] if dst_id in READ_TO_NODE else ANNOTATED_FASTA_DATA[dst_id][0]
    else:
        raise Exception("Unrecognised orientation pairing.")

    if src_id in READ_TO_NODE and dst_id in READ_TO_NODE:
        if src[1] == '+' and dst[1] == '+':
            src_n_id, dst_n_id = READ_TO_NODE[src_id][0], READ_TO_NODE[dst_id][0]
            src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][1], READ_TO_NODE[dst_rev[0]][1]
        elif src[1] == '+' and dst[1] == '-':
            src_n_id, dst_n_id = READ_TO_NODE[src_id][0], READ_TO_NODE[dst_id][1]
            src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][0], READ_TO_NODE[dst_rev[0]][1]
        elif src[1] == '-' and dst[1] == '+':
            src_n_id, dst_n_id = READ_TO_NODE[src_id][1], READ_TO_NODE[dst_id][0]
            src_rev_n_id, dst_rev_n_id = READ_TO_NODE[src_rev[0]][1], READ_TO_NODE[dst_rev[0]][0]

        if src_n_id == dst_n_id:
            # print("Self edge found!")
            return 0, data
        
        if dst_n_id in SUCCESSOR_DICT[src_n_id] or dst_rev_n_id in SUCCESSOR_DICT[src_rev_n_id]:
            # print("Edge already exists!")
            return 0, data
        
        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data = defaultdict(list)

        data['ol_similarity'].append(c_ol_similarity)
        data['ol_len'].append(c_ol_len)
        data['prefix_len'].append(c_prefix_len)
        data['valid_src'].append(src)
        data['valid_dst'].append(dst)

        data['ol_similarity'].append(c_ol_similarity)
        data['ol_len'].append(c_ol_len)
        data['prefix_len'].append(c_prefix_len_rev)
        data['valid_src'].append(src_rev)
        data['valid_dst'].append(dst_rev)

        return 1, data

    if src_id not in READ_TO_NODE:
        data = { 
            '+' : defaultdict(create_list_dd),
            '-' : defaultdict(create_list_dd)
        }

        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data[src[1]][src_id]['outs'].append(dst)
        data[src[1]][src_id]['ol_len_outs'].append(c_ol_len)
        data[src[1]][src_id]['ol_similarity_outs'].append(c_ol_similarity)   
        data[src[1]][src_id]['prefix_len_outs'].append(c_prefix_len)
        data[src[1]][src_id]['read_len'] = src_len

        data[dst_rev[1]][src_id]['ins'].append(src_rev)
        data[dst_rev[1]][src_id]['ol_len_ins'].append(c_ol_len)
        data[dst_rev[1]][src_id]['ol_similarity_ins'].append(c_ol_similarity)
        data[dst_rev[1]][src_id]['prefix_len_ins'].append(c_prefix_len_rev)
        data[dst_rev[1]][src_id]['read_len'] = src_len

        return 2, data

    if dst_id not in READ_TO_NODE:
        data = { 
            '+' : defaultdict(create_list_dd),
            '-' : defaultdict(create_list_dd)
        }

        c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
        edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
        c_ol_similarity = 1 - edit_dist / c_ol_len
        if src[0] == id1:
            src_len, dst_len = len1, len2
            c_prefix_len, c_prefix_len_rev = len1-c_ol_len, len2-c_ol_len
        else:
            src_len, dst_len = len2, len1
            c_prefix_len, c_prefix_len_rev = len2-c_ol_len, len1-c_ol_len

        data[dst[1]][dst_id]['ins'].append(src)
        data[dst[1]][dst_id]['ol_len_ins'].append(c_ol_len)
        data[dst[1]][dst_id]['ol_similarity_ins'].append(c_ol_similarity)
        data[dst[1]][dst_id]['prefix_len_ins'].append(c_prefix_len)
        data[dst[1]][dst_id]['read_len'] = dst_len

        data[src_rev[1]][dst_id]['outs'].append(dst_rev)
        data[src_rev[1]][dst_id]['ol_len_outs'].append(c_ol_len)
        data[src_rev[1]][dst_id]['ol_similarity_outs'].append(c_ol_similarity)
        data[src_rev[1]][dst_id]['prefix_len_outs'].append(c_prefix_len_rev)
        data[src_rev[1]][dst_id]['read_len'] = dst_len

        return 2, data

def parse_paf(paf_path, aux, name):
    '''
    ghosts = {
        '+' : {
            read_id : {
                'read_len' : Read length for this read
                'outs' : [read_id, ...]
                'ol_len_outs' : [ol_len, ...],
                'ol_similarity_outs' : [ol_similarity, ...],
                'prefix_len_outs' : [prefix_len, ...],
                'ins' : [read_id, ...],
                'ol_len_ins' : [ol_len, ...],
                'ol_similarity_ins' : [ol_similarity, ...],
                'prefix_len_ins' : [prefix_len, ...],
            }, 
            read_id_2 : { ... },
            ...
        },
        '-' : { ... }
    }
    '''
    print("Parsing paf file...")
    
    global READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ
    READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ = aux['read_seqs'], aux['read_to_node'], aux['annotated_fasta_data'], aux['successor_dict'], aux['node_to_read']

    with open(paf_path) as f:
        rows = f.readlines()

    rows = preprocess_rows(rows)

    valid_src, valid_dst, ol_len, ol_similarity, prefix_len = [], [], [], [], []
    rejected, ghosts = 0, {'+':defaultdict(create_list_dd), '-':defaultdict(create_list_dd)}
    nrows = len(rows)

    with Pool(20) as pool:
        results = pool.imap_unordered(parse_row, iter(rows), chunksize=60)
        for code, data in tqdm(results, total=nrows, ncols=120):
            if code == 0: 
                CASES[0] += 1
                continue
            elif code == 1:
                CASES[1] += 1
                ol_similarity.extend(data['ol_similarity'])
                ol_len.extend(data['ol_len'])
                prefix_len.extend(data['prefix_len'])
                valid_src.extend(data['valid_src']) 
                valid_dst.extend(data['valid_dst'])
            elif code == 2:
                CASES[2] += 1
                for orient, d in data.items():
                    for id, curr_data in d.items():
                        for label in ['outs', 'ol_len_outs', 'ol_similarity_outs', 'prefix_len_outs', 'ins', 'ol_len_ins', 'ol_similarity_ins', 'prefix_len_ins']:
                            ghosts[orient][id][label].extend(curr_data[label])

                        ghosts[orient][id]['read_len'] = curr_data['read_len']

    print("CASES:")
    for i, n in CASES.items():
        print("Case", i, ":", n)

    data = {
        'valid_src' : valid_src,
        'valid_dst' : valid_dst,
        'ol_len' : ol_len,
        'ol_similarity' : ol_similarity,
        'prefix_len' : prefix_len,
        'ghost_data' : ghosts
    }

    with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl", "wb") as p:
        pickle.dump(data, p)

    return data

def enhance_with_paf(g, aux, get_similarities=False, add_edges=True, add_node_features=True):
    '''
    This function adds all edges between two real nodes that were in paf but not in gfa. 
    For each real node, it also creates the following 6 new node features:
        1. Number of connected outgoing ghost nodes
        2. Average OL Len of edges connecting these outgoing ghost nodes
        3. Average OL Sim of edges connecting these outgoing ghost nodes
        4. Number of connected incoming ghost nodes
        5. Average OL Len of edges connecting these incoming ghost nodes
        6. Average OL Sim of edges connecting these incoming ghost nodes
    '''
    data = aux['paf_data']
    r2n = aux['read_to_node']
    valid_src, valid_dst, prefix_len, ol_len, ol_similarity = data['valid_src'], data['valid_dst'], data['prefix_len'], data['ol_len'], data['ol_similarity']

    if add_edges:
        print("Adding edges between real nodes...")
        old_n_edges = g.edge_index.size()[1]
        edge_index, overlap_length, prefix_length = deepcopy(g['edge_index']).tolist(), deepcopy(g['overlap_length']).tolist(), deepcopy(g['prefix_length']).tolist()
        if get_similarities: overlap_similarity = deepcopy(g['overlap_similarity']).tolist()
        
        for i in tqdm(range(len(valid_src)), ncols=120):
            # src and dst have format ('id', orientation)
            # r2n[x] has format (real_id, virt_id)
            src, dst = valid_src[i], valid_dst[i]
            if src[1] == '+' and dst[1] == '+':
                edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][0])
            elif src[1] == '-' and dst[1] == '-':
                edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][1])
            elif src[1] == '+' and dst[1] == '-':
                edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][1])
            elif src[1] == '-' and dst[1] == '+':
                edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][0])
            else:
                raise Exception("Unrecognised orientation pairing.")
            
            overlap_length.append(ol_len[i]); prefix_length.append(prefix_len[i])
            if get_similarities: overlap_similarity.append(ol_similarity[i])

        g['edge_index'] = torch.tensor(edge_index); g['overlap_length'] = torch.tensor(overlap_length); g['prefix_length'] = torch.tensor(prefix_length)
        g.E_ID = torch.cat((g.E_ID, torch.tensor([i for i in range(old_n_edges, g.edge_index.size()[1])])))
        if get_similarities: g['overlap_similarity'] = torch.tensor(overlap_similarity)

    if add_node_features:
        print("Adding node features...")
        ghosts = data['ghost_data']
        real_node_ghost_data = defaultdict(create_list_dd)

        for v in ghosts.values(): # '+' and '-'
            for ghost_info in tqdm(v.values(), ncols=120):
                for i, c_out in enumerate(ghost_info['outs']):
                    c_ol_len, c_ol_sim = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i]
                    if c_out[0] not in r2n: continue # this is not a valid node in the gfa

                    if c_out[1] == '+':
                        n_id = r2n[c_out[0]][0] 
                    else:
                        n_id = r2n[c_out[0]][1]

                    real_node_ghost_data[n_id]['ol_len_ins'].append(c_ol_len)
                    real_node_ghost_data[n_id]['ol_similarity_ins'].append(c_ol_sim)

                for i, c_in in enumerate(ghost_info['ins']):
                    c_ol_len, c_ol_sim = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i]
                    if c_in[0] not in r2n: continue # this is not a valid node in the gfa

                    if c_in[1] == '+':
                        n_id = r2n[c_in[0]][0] 
                    else:
                        n_id = r2n[c_in[0]][1]

                    real_node_ghost_data[n_id]['ol_len_outs'].append(c_ol_len)
                    real_node_ghost_data[n_id]['ol_similarity_outs'].append(c_ol_sim)

        n_outs, ol_len_outs, ol_similarity_outs, n_ins, ol_len_ins, ol_similarity_ins = torch.zeros(len(g.N_ID)), torch.zeros(len(g.N_ID)), torch.zeros(len(g.N_ID)), torch.zeros(len(g.N_ID)), torch.zeros(len(g.N_ID)), torch.zeros(len(g.N_ID))
        for ind, n_id in tqdm(enumerate(g.N_ID), ncols=120):
            c_ghost_data = real_node_ghost_data[int(n_id)]
            if c_ghost_data['ol_len_outs']:
                c_n_outs = len(c_ghost_data['ol_len_outs'])
                n_outs[ind] = c_n_outs
                ol_len_outs[ind] = sum(c_ghost_data['ol_len_outs'])/c_n_outs 
                ol_similarity_outs[ind] = sum(c_ghost_data['ol_similarity_outs'])/c_n_outs 

            if c_ghost_data['ol_len_ins']:
                c_n_ins = len(c_ghost_data['ol_len_ins'])
                n_ins[ind] = c_n_ins
                ol_len_ins[ind] = sum(c_ghost_data['ol_len_ins'])/c_n_ins
                ol_similarity_ins[ind] = sum(c_ghost_data['ol_similarity_ins'])/c_n_ins

        g['ghost_n_outs'] = n_outs
        g['ghost_ol_len_outs'] = ol_len_outs
        g['ghost_ol_sim_outs'] = ol_similarity_outs
        g['ghost_n_ins'] = n_ins
        g['ghost_ol_len_ins'] = ol_len_ins
        g['ghost_ol_sim_ins'] = ol_similarity_ins

    return g

def enhance_with_paf_2(g, aux, get_similarities=False, add_features=False):
    '''
    This function adds all edges between two real nodes that were in paf but not in gfa. 
    It also adds all ghost nodes and their respective edges in a 1-hop radius around the real graph.
    '''
    data = aux['paf_data']
    valid_src, valid_dst, prefix_len, ol_len, ol_similarity = data['valid_src'], data['valid_dst'], data['prefix_len'], data['ol_len'], data['ol_similarity']

    print("Adding edges between real nodes...")
    E_ID, N_ID = deepcopy(g.E_ID).tolist(), deepcopy(g.N_ID).tolist()
    c_n_id, c_e_id = len(N_ID), len(E_ID)
    edge_index, overlap_length, prefix_length, read_length = deepcopy(g['edge_index']).tolist(), deepcopy(g['overlap_length']).tolist(), deepcopy(g['prefix_length']).tolist(), deepcopy(g['read_length']).tolist()
    if get_similarities: overlap_similarity = deepcopy(g['overlap_similarity']).tolist()
    r2n = aux['read_to_node']
    for i in tqdm(range(len(valid_src)), ncols=120):
        # src and dst have format ('id', orientation)
        # r2n[x] has format (real_id, virt_id)
        src, dst = valid_src[i], valid_dst[i]

        if src[1] == '+' and dst[1] == '+':
            edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][0])
        elif src[1] == '-' and dst[1] == '-':
            edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][1])
        elif src[1] == '+' and dst[1] == '-':
            edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][1])
        elif src[1] == '-' and dst[1] == '+':
            edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][0])
        else:
            raise Exception("Unrecognised orientation pairing.")
        
        E_ID.append(c_e_id); c_e_id += 1
        overlap_length.append(ol_len[i]); prefix_length.append(prefix_len[i])
        if get_similarities: overlap_similarity.append(ol_similarity[i])

    print("Adding ghost nodes and edges...")
    ghosts = data['ghost_data']
    edges_added = set()
    fasta_data = aux['annotated_fasta_data']

    for read_id, ghost_info in tqdm(ghosts['+'].items(), ncols=120):
        added = 0
        for i, c_out in enumerate(ghost_info['outs']):
            c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i], ghost_info['prefix_len_outs'][i]
            if c_out[0] not in r2n: continue # this is not a valid node in the gfa

            if c_out[1] == '+':
                n_id = r2n[c_out[0]][0] 
            else:
                n_id = r2n[c_out[0]][1]

            if (c_n_id, n_id) in edges_added: continue

            edge_index[0].append(c_n_id)
            edge_index[1].append(n_id)
            overlap_length.append(c_ol_len)
            prefix_length.append(c_prefix_len)
            if get_similarities: overlap_similarity.append(c_ol_sim)
            edges_added.add((c_n_id, n_id))
            E_ID.append(c_e_id)
            c_e_id += 1
            added += 1

        for i, c_in in enumerate(ghost_info['ins']):
            c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i], ghost_info['prefix_len_ins'][i]
            if c_in[0] not in r2n: continue # this is not a valid node in the gfa

            if c_in[1] == '+':
                n_id = r2n[c_in[0]][0] 
            else:
                n_id = r2n[c_in[0]][1]

            if (n_id, c_n_id) in edges_added: continue

            edge_index[0].append(n_id)
            edge_index[1].append(c_n_id)
            overlap_length.append(c_ol_len)
            prefix_length.append(c_prefix_len)
            if get_similarities: overlap_similarity.append(c_ol_sim)
            edges_added.add((n_id, c_n_id))
            E_ID.append(c_e_id)
            c_e_id += 1
            added += 1

        if added > 0:
            aux['node_to_read'][c_n_id] = read_id
            aux['read_seqs'][c_n_id] = fasta_data[read_id][0]
            read_length.append(ghost_info['read_len'])
            N_ID.append(c_n_id)
            c_n_id += 1
        else:
            print("No edges added for this ghost node. That's weird...")
            continue

        ghost_info = ghosts['-'][read_id]
        if not ghost_info: raise ValueError("Missing reverse comp of ghost node!")
        added = 0

        for i, c_out in enumerate(ghost_info['outs']):
            c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i], ghost_info['prefix_len_outs'][i]
            if c_out[0] not in r2n: continue # this is not a valid node in the gfa

            if c_out[1] == '+':
                n_id = r2n[c_out[0]][0] 
            else:
                n_id = r2n[c_out[0]][1]

            if (c_n_id, n_id) in edges_added: continue

            edge_index[0].append(c_n_id)
            edge_index[1].append(n_id)
            overlap_length.append(c_ol_len)
            prefix_length.append(c_prefix_len)
            if get_similarities: overlap_similarity.append(c_ol_sim)
            edges_added.add((c_n_id, n_id))
            E_ID.append(c_e_id)
            c_e_id += 1
            added += 1

        for i, c_in in enumerate(ghost_info['ins']):
            c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i], ghost_info['prefix_len_ins'][i]
            if c_in[0] not in r2n: continue # this is not a valid node in the gfa

            if c_in[1] == '+':
                n_id = r2n[c_in[0]][0] 
            else:
                n_id = r2n[c_in[0]][1]

            if (n_id, c_n_id) in edges_added: continue

            edge_index[0].append(n_id)
            edge_index[1].append(c_n_id)
            overlap_length.append(c_ol_len)
            prefix_length.append(c_prefix_len)
            if get_similarities: overlap_similarity.append(c_ol_sim)
            edges_added.add((n_id, c_n_id))
            E_ID.append(c_e_id)
            c_e_id += 1
            added += 1

        if added > 0:
            aux['node_to_read'][c_n_id] = read_id
            aux['read_seqs'][c_n_id] = fasta_data[read_id][1]
            read_length.append(ghost_info['read_len'])
            N_ID.append(c_n_id)
            c_n_id += 1
        else:
            print("No edges added for this reverse comp ghost node. That's weird...")

    edges_added, nodes_added = len(E_ID)-g.E_ID.size()[0], len(N_ID)-g.N_ID.size()[0]
    print("Nodes added:", nodes_added, "Edges added:", edges_added)

    if add_features: 
        is_real_edge, is_real_node = [1]*g.E_ID.size()[0], [1]*g.N_ID.size()[0]
        is_real_edge.extend([0]*edges_added); is_real_node.extend([0]*nodes_added)
        g.is_real_edge = is_real_edge; g.is_real_node = is_real_node

    g['edge_index'] = torch.tensor(edge_index); g['overlap_length'] = torch.tensor(overlap_length); g['prefix_length'] = torch.tensor(prefix_length); g['read_length'] = torch.tensor(read_length)
    g.E_ID = torch.tensor(E_ID); g.N_ID = torch.tensor(N_ID)
    if get_similarities: g['overlap_similarity'] = torch.tensor(overlap_similarity)

    return g, aux

def check_duplicate_edges(g):
    _, inv_indices = g.edge_index.t().contiguous().unique(return_inverse=True, dim=0)
    duplicate_edges_count = inv_indices.bincount()
    duplicate_edges_mask = duplicate_edges_count[inv_indices] > 1

    # Ensure the mask is correctly aligned
    duplicate_edges = g.edge_index[:, duplicate_edges_mask]

    # Print duplicate edges
    print("Duplicate edges:")
    for i in range(duplicate_edges.shape[1]):
        print(f"Edge from {duplicate_edges[0, i]} to {duplicate_edges[1, i]}")

