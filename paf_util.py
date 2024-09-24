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

READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ, READS_PARSED = None, None, None, None, None, set()

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

                # Overlaps where edge is already in gfa
                if dst_n_id in SUCCESSOR_DICT[src_n_id] or dst_rev_n_id in SUCCESSOR_DICT[src_rev_n_id]:
                    duplicates += 1
                    continue

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

    if not READ_SEQS or not READ_TO_NODE or not ANNOTATED_FASTA_DATA or not SUCCESSOR_DICT or not READS_PARSED:
        raise ValueError("Global objects not set!")

    row_split = row.strip().split()

    ## What are these last 3 fields? ##
    id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row_split
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
    
    if str(src_id) not in READS_PARSED and str(dst_id) not in READS_PARSED: 
        return 3, row
        
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

    if str(src_id) in READS_PARSED and str(dst_id) in READS_PARSED:
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

    if str(src_id) not in READS_PARSED:
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

    if str(dst_id) not in READS_PARSED:
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
        'hop_<n>' {
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
        },
        'hop_<n+1>' : { ... }
    }
    '''
    print("Parsing paf file...")
    
    global READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ, READS_PARSED
    READ_SEQS, READ_TO_NODE, ANNOTATED_FASTA_DATA, SUCCESSOR_DICT, NODE_TO_READ, READS_PARSED = aux['read_seqs'], aux['read_to_node'], aux['annotated_fasta_data'], aux['successor_dict'], aux['node_to_read'], set()

    for c_n_id in sorted(NODE_TO_READ.keys()):
        if c_n_id % 2 != 0: continue # Skip all virtual nodes
        read_id = NODE_TO_READ[c_n_id]
        if isinstance(read_id, list):
            READS_PARSED.add(read_id[0][0]); READS_PARSED.add(read_id[-1][0])
        else:
            READS_PARSED.add(read_id)

    with open(paf_path) as f:
        rows = f.readlines()

    rows = preprocess_rows(rows)
    curr_rows = deepcopy(rows)
    cutoff = len(curr_rows) * 0.01

    valid_src, valid_dst, ol_len, ol_similarity, prefix_len, edge_hops = [], [], [], [], [], []
    ghosts = {}
    
    next_rows, hop = [], 1

    while len(curr_rows) > cutoff:
        print(f"Starting run for Hop {hop}. nrows: {len(curr_rows)}, cutoff: {cutoff}")
        curr_ghost_info = {'+':defaultdict(create_list_dd), '-':defaultdict(create_list_dd)}

        with Pool(40) as pool:
            results = pool.imap_unordered(parse_row, iter(curr_rows), chunksize=160)
            for code, data in tqdm(results, total=len(curr_rows), ncols=120):
                if code == 0: 
                    continue
                elif code == 1:
                    ol_similarity.extend(data['ol_similarity'])
                    ol_len.extend(data['ol_len'])
                    prefix_len.extend(data['prefix_len'])
                    valid_src.extend(data['valid_src']) 
                    valid_dst.extend(data['valid_dst'])
                    edge_hops.extend([hop]*len(data['valid_src']))
                elif code == 2:
                    for orient, d in data.items():
                        for id, curr_data in d.items():
                            for label in ['outs', 'ol_len_outs', 'ol_similarity_outs', 'prefix_len_outs', 'ins', 'ol_len_ins', 'ol_similarity_ins', 'prefix_len_ins']:
                                curr_ghost_info[orient][id][label].extend(curr_data[label])

                            curr_ghost_info[orient][id]['read_len'] = curr_data['read_len']
                elif code == 3:
                    next_rows.append(data)

        assert set(curr_ghost_info['+'].keys()) == set(curr_ghost_info['-'].keys()), "Missing real-virtual node pair."
        for read_id in curr_ghost_info['+'].keys():
            READS_PARSED.add(str(read_id))

        print(f"Finished run for Hop {hop}. nrows in hop: {len(curr_rows) - len(next_rows)}")
        ghosts['hop_'+str(hop)] = curr_ghost_info
        curr_rows = deepcopy(next_rows)
        next_rows = []
        hop += 1

    data = {
        'valid_src' : valid_src,
        'valid_dst' : valid_dst,
        'ol_len' : ol_len,
        'ol_similarity' : ol_similarity,
        'prefix_len' : prefix_len,
        'edge_hops' : edge_hops,
        'ghost_data' : ghosts
    }

    with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl", "wb") as p:
        pickle.dump(data, p)

    return data

def enhance_with_paf_2(g, aux, get_similarities=False, hop=1, add_inner_edges=True):
    E_ID, N_ID = deepcopy(g.E_ID).tolist(), deepcopy(g.N_ID).tolist()
    c_n_id, c_e_id = len(N_ID), len(E_ID)
    edge_index, overlap_length, prefix_length, read_length = deepcopy(g['edge_index']).tolist(), deepcopy(g['overlap_length']).tolist(), deepcopy(g['prefix_length']).tolist(), deepcopy(g['read_length']).tolist()
    if get_similarities: overlap_similarity = deepcopy(g['overlap_similarity']).tolist()
    node_hop, edge_hop = [0] * c_n_id, [0] * c_e_id

    r2n = aux['read_to_node']
    data = aux['paf_data']

    print("Adding ghost nodes and edges...")
    ghosts = data['ghost_data']
    fasta_data = aux['annotated_fasta_data']

    edges_added = set()
    for i, src in enumerate(edge_index[0]):
        edges_added.add((int(src), int(edge_index[1][i])))
    dups_caught = { i:0 for i in range(5) }

    for curr_hop in range(1, hop+1):
        name = 'hop_' + str(curr_hop)
        if name not in ghosts: break # No more ghost data

        curr_ghost_data = ghosts[name]
        for read_id, ghost_info in tqdm(curr_ghost_data['+'].items(), ncols=120):
            added = 0
            for i, c_out in enumerate(ghost_info['outs']):
                c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i], ghost_info['prefix_len_outs'][i]
                if c_out[0] not in r2n: continue # this is not a valid node in the graph

                if c_out[1] == '+':
                    n_id = r2n[c_out[0]][0] 
                else:
                    n_id = r2n[c_out[0]][1]

                if (c_n_id, n_id) in edges_added: 
                    dups_caught[0] += 1
                    continue

                edge_index[0].append(c_n_id)
                edge_index[1].append(n_id)
                overlap_length.append(c_ol_len)
                prefix_length.append(c_prefix_len)
                if get_similarities: overlap_similarity.append(c_ol_sim)
                edges_added.add((c_n_id, n_id))
                E_ID.append(c_e_id)
                edge_hop.append(curr_hop)
                c_e_id += 1
                added += 1

            for i, c_in in enumerate(ghost_info['ins']):
                c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i], ghost_info['prefix_len_ins'][i]
                if c_in[0] not in r2n: continue # this is not a valid node in the graph

                if c_in[1] == '+':
                    n_id = r2n[c_in[0]][0] 
                else:
                    n_id = r2n[c_in[0]][1]

                if (n_id, c_n_id) in edges_added:
                    dups_caught[1] += 1
                    continue

                edge_index[0].append(n_id)
                edge_index[1].append(c_n_id)
                overlap_length.append(c_ol_len)
                prefix_length.append(c_prefix_len)
                if get_similarities: overlap_similarity.append(c_ol_sim)
                edges_added.add((n_id, c_n_id))
                E_ID.append(c_e_id)
                edge_hop.append(curr_hop)
                c_e_id += 1
                added += 1

            if added > 0:
                aux['node_to_read'][c_n_id] = read_id
                aux['read_seqs'][c_n_id] = fasta_data[read_id][0]
                read_length.append(ghost_info['read_len'])
                N_ID.append(c_n_id)
                node_hop.append(curr_hop)
                c_n_id += 1
            else:
                print("No edges added for this ghost node. That's weird...")
                continue

            ghost_info = curr_ghost_data['-'][read_id]
            if not ghost_info: raise ValueError("Missing reverse comp of ghost node!")
            added = 0

            for i, c_out in enumerate(ghost_info['outs']):
                c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i], ghost_info['prefix_len_outs'][i]
                if c_out[0] not in r2n: continue # this is not a valid node in the graph

                if c_out[1] == '+':
                    n_id = r2n[c_out[0]][0] 
                else:
                    n_id = r2n[c_out[0]][1]

                if (c_n_id, n_id) in edges_added:
                    dups_caught[2] += 1
                    continue

                edge_index[0].append(c_n_id)
                edge_index[1].append(n_id)
                overlap_length.append(c_ol_len)
                prefix_length.append(c_prefix_len)
                if get_similarities: overlap_similarity.append(c_ol_sim)
                edges_added.add((c_n_id, n_id))
                E_ID.append(c_e_id)
                edge_hop.append(curr_hop)
                c_e_id += 1
                added += 1

            for i, c_in in enumerate(ghost_info['ins']):
                c_ol_len, c_ol_sim, c_prefix_len = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i], ghost_info['prefix_len_ins'][i]
                if c_in[0] not in r2n: continue # this is not a valid node in the graph

                if c_in[1] == '+':
                    n_id = r2n[c_in[0]][0] 
                else:
                    n_id = r2n[c_in[0]][1]

                if (n_id, c_n_id) in edges_added:
                    dups_caught[3] += 1
                    continue

                edge_index[0].append(n_id)
                edge_index[1].append(c_n_id)
                overlap_length.append(c_ol_len)
                prefix_length.append(c_prefix_len)
                if get_similarities: overlap_similarity.append(c_ol_sim)
                edges_added.add((n_id, c_n_id))
                E_ID.append(c_e_id)
                edge_hop.append(curr_hop)
                c_e_id += 1
                added += 1

            if added > 0:
                aux['node_to_read'][c_n_id] = read_id
                aux['read_seqs'][c_n_id] = fasta_data[read_id][1]
                read_length.append(ghost_info['read_len'])
                N_ID.append(c_n_id)
                node_hop.append(curr_hop)
                c_n_id += 1
            else:
                print("No edges added for this reverse comp ghost node. That's weird...")

            r2n[read_id] = (c_n_id-2, c_n_id-1)


    if add_inner_edges:
        print("Adding more edges between nodes...")
        valid_src, valid_dst, prefix_len, ol_len, ol_similarity, edge_hops = data['valid_src'], data['valid_dst'], data['prefix_len'], data['ol_len'], data['ol_similarity'], data['edge_hops']

        for i in tqdm(range(len(valid_src)), ncols=120):
            if edge_hops[i] > max(1, hop): continue

            # src and dst have format ('id', orientation)
            # r2n[x] has format (real_id, virt_id)
            src, dst = valid_src[i], valid_dst[i]

            if src[1] == '+' and dst[1] == '+':
                src_n_id, dst_n_id = r2n[src[0]][0], r2n[dst[0]][0]
                # edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][0])
            elif src[1] == '-' and dst[1] == '-':
                src_n_id, dst_n_id = r2n[src[0]][1], r2n[dst[0]][1]
                # edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][1])
            elif src[1] == '+' and dst[1] == '-':
                src_n_id, dst_n_id = r2n[src[0]][0], r2n[dst[0]][1]
                # edge_index[0].append(r2n[src[0]][0]); edge_index[1].append(r2n[dst[0]][1])
            elif src[1] == '-' and dst[1] == '+':
                src_n_id, dst_n_id = r2n[src[0]][1], r2n[dst[0]][0]
                # edge_index[0].append(r2n[src[0]][1]); edge_index[1].append(r2n[dst[0]][0])
            else:
                raise Exception("Unrecognised orientation pairing.")

            if (src_n_id, dst_n_id) in edges_added:
                dups_caught[4] += 1
                continue
            
            edges_added.add((src_n_id, dst_n_id))

            edge_index[0].append(src_n_id); edge_index[1].append(dst_n_id)
            E_ID.append(c_e_id); c_e_id += 1
            edge_hop.append(edge_hops[i])
            overlap_length.append(ol_len[i]); prefix_length.append(prefix_len[i])
            if get_similarities: overlap_similarity.append(ol_similarity[i])


    edges_added, nodes_added = len(E_ID)-g.E_ID.size()[0], len(N_ID)-g.N_ID.size()[0]
    print("Nodes added:", nodes_added, "Edges added:", edges_added)
    print("Duplicates caught:")
    for case, val in dups_caught.items():
        print("Case:", case, "Val:", val)

    g.node_hop = torch.tensor(node_hop); g.edge_hop = torch.tensor(edge_hop)

    g['edge_index'] = torch.tensor(edge_index); g['overlap_length'] = torch.tensor(overlap_length); g['prefix_length'] = torch.tensor(prefix_length); g['read_length'] = torch.tensor(read_length)
    g.E_ID = torch.tensor(E_ID); g.N_ID = torch.tensor(N_ID)
    if get_similarities: g['overlap_similarity'] = torch.tensor(overlap_similarity)
    aux['read_to_node'] = r2n

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

