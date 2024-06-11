from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import edlib
import torch
import seaborn as sns
from tqdm import tqdm

# We have to do this because cannot pickle defaultdicts created by lambda
def create_list_dd():
    return defaultdict(list)

def parse_paf(paf_path, aux, name):
    '''
    ghosts = {
        '+' : {
            read_id : {
                'outs' : [read_id, ...]
                'ol_len_outs' : [ol_len, ...],
                'ol_similarity_outs' : [ol_similarity, ...],
                'ins' : [read_id, ...]
                'ol_len_ins' : [ol_len, ...],
                'ol_similarity_ins' : [ol_similarity, ...],
            }, 
            read_id_2 : { ... },
            ...
        },
        '-' : { ... }
    }
    '''
    print("Parsing paf file...")
    with open(paf_path) as f:
        rows = f.readlines()

    valid_src, valid_dst, ol_len, ol_similarity = [], [], [], []
    rejected, ghosts = 0, {'+':defaultdict(create_list_dd), '-':defaultdict(create_list_dd)}
    for row in tqdm(rows, ncols=120):
        row = row.strip().split()

        ## What are these last 3 fields? ##
        id1, len1, start1, end1, orientation, id2, len2, start2, end2, _, _, _ = row

        is_valid, src, dst = False, None, None
        if orientation == '+':
            if start1 == 0 and start2 == 0:
                rejected += 1
                pass
            elif end1 == len1 and end2 == len2:
                rejected += 1
                pass
            elif end1 == len1 and start2 == 0:
                src, dst = (id1, '+'), (id2, '+')
                src_rev, dst_rev = (id2, '-'), (id1, '-')
                is_valid = True
            elif start1 == 0 and end2 == len2:
                src, dst = (id2, '+'), (id1, '+')
                src_rev, dst_rev = (id1, '-'), (id2, '-')
                is_valid = True 
        else:
            if start1 == 0 and end2 == len2:
                rejected += 1
                pass
            elif end1 == len1 and start2 == 0:
                rejected += 1
                pass
            elif end1 == len1 and end2 == len2:
                src, dst = (id1, '+'), (id2, '-')
                src_rev, dst_rev = (id2, '+'), (id1, '-')
                is_valid = True
            elif start1 == 0 and start2 == 0:
                src, dst = (id1, '-'), (id2, '+')
                src_rev, dst_rev = (id2, '-'), (id1, '+')
                is_valid = True

        if is_valid:
            start1, end1, start2, end2 = int(start1), int(end1), int(start2), int(end2)
            read_seqs, read_to_node, annotated_fasta_data = aux['read_seqs'], aux['read_to_node'], aux['annotated_fasta_data']

            src_id, dst_id = src[0], dst[0]
            if src[1] == '+' and dst[1] == '+':
                src_seq = read_seqs[read_to_node[src_id][0]] if src_id in read_to_node else annotated_fasta_data[src_id][0]
                dst_seq = read_seqs[read_to_node[dst_id][0]] if dst_id in read_to_node else annotated_fasta_data[dst_id][0]
            elif src[1] == '+' and dst[1] == '-':
                src_seq = read_seqs[read_to_node[src_id][0]] if src_id in read_to_node else annotated_fasta_data[src_id][0]
                dst_seq = read_seqs[read_to_node[dst_id][1]] if dst_id in read_to_node else annotated_fasta_data[dst_id][1]
            elif src[1] == '-' and dst[1] == '+':
                src_seq = read_seqs[read_to_node[src_id][1]] if src_id in read_to_node else annotated_fasta_data[src_id][1]
                dst_seq = read_seqs[read_to_node[dst_id][0]] if dst_id in read_to_node else annotated_fasta_data[dst_id][0]
            else:
                raise Exception("Unrecognised orientation pairing.")

            c_ol_len = end1-start1 # overlapping region length might not always be equal between source and target. but we always take source for ol length
            edit_dist = edlib.align(src_seq, dst_seq)['editDistance']
            c_ol_similarity = 1 - edit_dist / c_ol_len

            if src_id in read_to_node and dst_id in read_to_node:
                ol_similarity.extend([c_ol_similarity, c_ol_similarity])
                ol_len.extend([c_ol_len, c_ol_len])
                valid_src.extend([src, src_rev]) 
                valid_dst.extend([dst, dst_rev])

            if src_id not in read_to_node:
                ghosts[src[1]][src_id]['outs'].append(dst)
                ghosts[src[1]][src_id]['ol_len_outs'].append(c_ol_len)
                ghosts[src[1]][src_id]['ol_similarity_outs'].append(c_ol_similarity)   

                ghosts[dst_rev[1]][src_id]['ins'].append(src_rev)
                ghosts[dst_rev[1]][src_id]['ol_len_ins'].append(c_ol_len)
                ghosts[dst_rev[1]][src_id]['ol_similarity_ins'].append(c_ol_similarity)

            if dst_id not in read_to_node:
                ghosts[dst[1]][dst_id]['ins'].append(src)
                ghosts[dst[1]][dst_id]['ol_len_ins'].append(c_ol_len)
                ghosts[dst[1]][dst_id]['ol_similarity_ins'].append(c_ol_similarity)

                ghosts[src_rev[1]][dst_id]['outs'].append(dst_rev)
                ghosts[src_rev[1]][dst_id]['ol_len_outs'].append(c_ol_len)
                ghosts[src_rev[1]][dst_id]['ol_similarity_outs'].append(c_ol_similarity)


        # if rejected % 10 == 0: print("rejected! len1:", len1, " start1: ", start1, " end1:", end1, " len2:", len2, " start2:", start2, " end2:", end2)

    print("accepted count:", len(valid_src), "rejected count:", rejected, "missing nodes count:", len(ghosts['+'])+len(ghosts['-']))

    data = {
        'valid_src' : valid_src,
        'valid_dst' : valid_dst,
        'ol_len' : ol_len,
        'ol_similarity' : ol_similarity,
        'ghost_data' : ghosts
    }

    with open(f"static/pkl/{name}_paf_data.pkl", "wb") as p:
        pickle.dump(data, p)

    return data

def enhance_with_paf(g, aux, name, get_similarities=False):
    data = aux['paf_data']
    valid_src, valid_dst, ol_len, ol_similarity = data['valid_src'], data['valid_dst'], data['ol_len'], data['ol_similarity']

    print("Enhancing graph...")
    edge_index, overlap_length = deepcopy(g['edge_index']).tolist(), deepcopy(g['overlap_length']).tolist()
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
        
        overlap_length.append(ol_len[i])
        if get_similarities: overlap_similarity.append(ol_similarity[i])

    g['edge_index'] = torch.tensor(edge_index); g['overlap_length'] = torch.tensor(overlap_length); 
    if get_similarities: g['overlap_similarity'] = torch.tensor(overlap_similarity)

    print("Parsing ghost node data...")
    ghosts = data['ghost_data']
    real_node_ghost_data = defaultdict(create_list_dd)
    for v in ghosts.values(): # '+' and '-'
        for ghost_info in tqdm(v.values(), ncols=120):
            for i, c_out in enumerate(ghost_info['outs']):
                c_ol_len, c_ol_sim = ghost_info['ol_len_outs'][i], ghost_info['ol_similarity_outs'][i]
                if c_out[0] not in r2n: continue # this is not a valid node in the gfa
                # print('node found out!')

                if c_out[1] == '+':
                    n_id = r2n[c_out[0]][0] 
                else:
                    n_id = r2n[c_out[0]][1]

                real_node_ghost_data[n_id]['ol_len_ins'].append(c_ol_len)
                real_node_ghost_data[n_id]['ol_similarity_ins'].append(c_ol_sim)

            for i, c_in in enumerate(ghost_info['ins']):
                c_ol_len, c_ol_sim = ghost_info['ol_len_ins'][i], ghost_info['ol_similarity_ins'][i]
                if c_in[0] not in r2n: continue # this is not a valid node in the gfa
                # print('node found in!')

                if c_in[1] == '+':
                    n_id = r2n[c_in[0]][0] 
                else:
                    n_id = r2n[c_in[0]][1]

                real_node_ghost_data[n_id]['ol_len_outs'].append(c_ol_len)
                real_node_ghost_data[n_id]['ol_similarity_outs'].append(c_ol_sim)

    print("Adding ghost node data...")
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

    torch.save(g, f'static/graphs/{name}_ghost.pt')

    return g

def analyse(data, name):
    print("Analysing ghost node data...")
    ghost_copy = {}
    for k, v in data['ghost_data']['+'].items():
        ghost_copy[k+'+'] = v
    for k, v in data['ghost_data']['-'].items():
        ghost_copy[k+'-'] = v
    
    for val_dict in tqdm(ghost_copy.values(), ncols=120):
        val_dict['out_count'] = len(val_dict['outs'])
        val_dict['ol_len_out_avg'] = 0 if val_dict['out_count'] == 0 else sum(val_dict['ol_len_outs'])/val_dict['out_count']
        val_dict['ol_similarity_out_avg'] = 0 if val_dict['out_count'] == 0 else sum(val_dict['ol_similarity_outs'])/val_dict['out_count']

        val_dict['in_count'] = len(val_dict['ins'])
        val_dict['ol_len_in_avg'] = 0 if val_dict['in_count'] == 0 else sum(val_dict['ol_len_ins'])/val_dict['in_count']
        val_dict['ol_similarity_in_avg'] = 0 if val_dict['in_count'] == 0 else sum(val_dict['ol_similarity_ins'])/val_dict['in_count']

    df = pd.DataFrame(ghost_copy).T
    df = df[df['out_count'] != 0]
    df = df.sort_values(by='out_count')

    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(df.index, df['ol_len_out_avg'], label='OL Len Out')
    ax.set_xlabel('ID'); ax.set_ylabel('Val'); ax.legend()
    plt.savefig(f'static/fig/{name}_ghost_ollen_out.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(df.index, df['ol_similarity_out_avg'], label='OL Similarity Out')
    ax.set_xlabel('ID'); ax.set_ylabel('Val'); ax.legend()
    plt.savefig(f'static/fig/{name}_ghost_olsim_out.png')
    plt.close()

    # IN GRAPHS WILL BE THE SAME AS OUT GRAPHS

    # df = pd.DataFrame(ghost_copy).T
    # df = df[df['in_count'] != 0]
    # df = df.sort_values(by='in_count')

    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.plot(df.index, df['ol_len_in_avg'], label='OL Len In')
    # ax.set_xlabel('ID'); ax.set_ylabel('Val'); ax.legend()
    # plt.savefig(f'static/fig/{name}_ghost_ollen_in.png')

    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.plot(df.index, df['ol_similarity_in_avg'], label='OL Similarity In')
    # ax.set_xlabel('ID'); ax.set_ylabel('Val'); ax.legend()
    # plt.savefig(f'static/fig/{name}_ghost_olsim_in.png')

def analyse2(g, name):
    print("Analysing real node ghost data...")
    df = pd.DataFrame({ 'count' : g.ghost_n_outs.tolist(), 'ol_len_outs' : g.ghost_ol_len_outs.tolist(), 'ol_sim_outs' : g.ghost_ol_sim_outs.tolist() })
    df = df.sort_values('count')

    sns.lineplot(data=df, x='count', y='ol_len_outs')
    plt.grid(True)
    plt.savefig(f'static/fig/{name}/{name}_rn_ghost_ol_len_outs.png')
    plt.close()
    sns.lineplot(data=df, x='count', y='ol_sim_outs')
    plt.grid(True)
    plt.savefig(f'static/fig/{name}/{name}_rn_ghost_ol_sim_outs.png')
    plt.close()

    # IN GRAPHS WILL BE THE SAME AS OUT GRAPHS

    # df = pd.DataFrame({ 'count' : g.ghost_n_ins.tolist(), 'ol_len_ins' : g.ghost_ol_len_ins.tolist(), 'ol_sim_ins' : g.ghost_ol_sim_ins.tolist() })
    # df = df.sort_values('count')
    # sns.lineplot(data=df, x='count', y='ol_len_ins')
    # plt.grid(True)
    # plt.savefig(f'static/fig/{name}/{name}_rn_ghost_ol_len_ins.png')
    # plt.close()
    # sns.lineplot(data=df, x='count', y='ol_sim_ins')
    # plt.grid(True)
    # plt.savefig(f'static/fig/{name}/{name}_rn_ghost_ol_sim_ins.png')
    # plt.close()

