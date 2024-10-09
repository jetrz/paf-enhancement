import argparse, dgl, math, os, pickle, subprocess
from Bio import Seq, SeqIO
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# For telomere operations
REP1, REP2 = 'TTAGGG', 'CCCTAA' # Repetitive regions used for identifying telomeric sequences
CHOP_WALK_BUFFER, CHOP_WALK_MIN_COUNT_DEFAULT = 10, 100 # allowed buffer between reps, default number of times the target sequence must repeat for it to qualify as telomeric region
# For use in cutoff metric
OL_LEN_CUTOFF_50, OL_LEN_CUTOFF_75, OL_LEN_CUTOFF_90 = 3875, 6765, 10000 # 50th, 75th, 90th percentile of OL Len for edges with gt_bin=0, averaged over training graphs. For use in cutoff metric

# REP_THRESHOLD = 0.05 # % of seq that must contain the above rep regions for it to be classified as telomeric
# REP_THRESHOLD = 0.005
# SEARCH_REGION_P, MAX_SEARCH_REGION_LEN = 0.1, 30000 # Size of start and end region of walk to search

class Edge():
    def __init__(self, new_src_nid, new_dst_nid, old_src_nid, old_dst_nid, prefix_len, ol_len, ol_sim):
        self.new_src_nid = new_src_nid
        self.new_dst_nid = new_dst_nid
        self.old_src_nid = old_src_nid
        self.old_dst_nid = old_dst_nid
        self.prefix_len = prefix_len
        self.ol_len = ol_len
        self.ol_sim = ol_sim

class AdjList():
    """
    Maps new_src_nid to edges.
    """

    def __init__(self):
        self.adj_list = defaultdict(set)

    def add_edge(self, edge):
        self.adj_list[edge.new_src_nid].add(edge)

    def remove_edge(self, edge):
        neighbours = self.adj_list[edge.new_src_nid]
        if edge not in neighbours:
            print("WARNING: Removing an edge that does not exist!")
        self.adj_list[edge.new_src_nid].discard(edge)
        if not self.adj_list[edge.new_src_nid]: del self.adj_list[edge.new_src_nid]

    def get_edge(self, new_src_nid, new_dst_nid):
        for e in self.adj_list[new_src_nid]:
            if e.new_dst_nid == new_dst_nid: 
                return e
            
    def remove_node(self, n_id):
        if n_id in self.adj_list: del self.adj_list[n_id]

        new_adj_list = defaultdict(set)
        for new_src_nid, neighbours in self.adj_list.items():
            new_neighbours = set(e for e in neighbours if e.new_dst_nid != n_id)
            if new_neighbours: new_adj_list[new_src_nid] = new_neighbours
        self.adj_list = new_adj_list

    def get_neighbours(self, n_id):
        return self.adj_list.get(n_id, [])
    
    def __str__(self):
        n_nodes, n_edges = len(self.adj_list), sum(len(v) for v in self.adj_list.values())
        text = f"Number of nodes: {n_nodes}, Number of edges: {n_edges}\n"
        for k, v in self.adj_list.items():
            c_text = f"Node: {k}, Neighbours: "
            for e in v:
                c_text += f"{e.new_dst_nid}, "
            text += c_text[:-2]
            text += "\n"
        return text

def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'

def chop_walks(old_walks, n2s, chop_walk_min_count):
    def chop_indiv_walk(walk):
        # This function could probably be optimised by only using 2 or even 1 pass through the nodes. But to minimise bugs i'm doing it step by step
        seq, curr_pos, pos_to_node = "", 0, {}
        for node in walk:
            # Preprocess the sequence
            c_seq = str(n2s[node])
            seq += c_seq            
            c_len_seq = len(c_seq)
            for i in range(curr_pos, curr_pos+c_len_seq):
                pos_to_node[i] = node
            curr_pos += c_len_seq
            
        # Generate telomeric regions
        i = 0
        rep_regs = {}
        while i < len(seq):
            curr_telo = None
            if seq.startswith(REP1, i) or seq.startswith(REP2, i):
                start_node = pos_to_node[i]
                if seq.startswith(REP1, i):
                    curr_telo = REP1
                elif seq.startswith(REP2, i):
                    curr_telo = REP2

                curr_count = 1
                end_index = i+6
                while end_index < len(seq):
                    next_index = seq.find(curr_telo, end_index)
                    if next_index == -1: break
                    if next_index - end_index <= CHOP_WALK_BUFFER:
                        curr_count += 1
                        end_index = next_index + 6
                    else:
                        break

                i = end_index

                if curr_count >= chop_walk_min_count:
                    rep_regs[start_node] = (pos_to_node[i-1], curr_telo)
            else:
                i += 1

        # Chop walks
        new_walks, telo_ref = [], {}
        curr_ind, curr_walk, curr_telo = 0, [], None
        while curr_ind < len(walk):
            curr_node = walk[curr_ind]
            if curr_node in rep_regs:
                end_node, telo_type = rep_regs[curr_node]
                if curr_telo is None:
                    curr_telo = telo_type
                    init_walk_len = len(curr_walk)
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: break
                    if len(curr_walk) > 2*init_walk_len: # if the telomeric region is as long as the walk preceding it, chop it off
                        new_walks.append(curr_walk.copy())
                        telo_ref[len(new_walks)-1] = {
                            'start' : None,
                            'end' : '+' if curr_telo == REP1 else '-'
                        }
                        curr_walk, curr_telo = [], None
                elif curr_telo != telo_type:
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: 
                            new_walks.append(curr_walk.copy())
                            if curr_telo == REP1:
                                telo_ref[len(new_walks)-1] = {
                                    'start' : '+',
                                    'end' : '-'
                                }
                            else:
                                telo_ref[len(new_walks)-1] = {
                                    'start' : '-',
                                    'end' : '+'
                                }
                            curr_walk, curr_telo = [], None
                            break
                else:
                    new_walks.append(curr_walk.copy())
                    telo_ref[len(new_walks)-1] = {
                        'start' : '+' if curr_telo == REP1 else '-',
                        'end' : None
                    }
                    curr_walk, curr_telo = [], telo_type
                    while True:
                        curr_node = walk[curr_ind]
                        curr_walk.append(curr_node)
                        curr_ind += 1
                        if curr_node == end_node: break
            else:
                curr_walk.append(curr_node)
                curr_ind += 1

        if curr_walk: 
            new_walks.append(curr_walk.copy())
            if curr_telo == REP1:
                start_telo = '+'
            elif curr_telo == REP2:
                start_telo = '-'
            else:
                start_telo = None
            telo_ref[len(new_walks)-1] = {
                'start' : start_telo,
                'end' : None
            }

        # Sanity Check
        combined_list = [item for inner in new_walks for item in inner]
        assert combined_list == walk, "Not all nodes accounted for when chopping old walk!"

        return new_walks, telo_ref

    new_walks, telo_ref = [], {}
    for walk in tqdm(old_walks, ncols=120):
        c_walks, c_telo_ref = chop_indiv_walk(walk)
        for i, w in enumerate(c_walks):
            new_walks.append(w)
            telo_ref[len(new_walks)-1] = c_telo_ref[i]

    print(f"Chopping complete! n Old Walks: {len(old_walks)}, n New Walks: {len(new_walks)}")
    return new_walks, telo_ref

def get_best_walk(adj_list, start_node, n_old_walks, telo_ref, penalty=None, memo_chances=50):
    # Dictionary to memoize the longest walk from each node
    memo, memo_counts = {}, defaultdict(int)

    def dfs(node, visited, walk_telo):
        if node < n_old_walks:
            if telo_ref[node]['start']:
                if walk_telo: print("WARNING: Trying to set walk_telo when it is already set!")
                walk_telo = telo_ref[node]['start']
            elif telo_ref[node]['end']:
                if walk_telo: print("WARNING: Trying to set walk_telo when it is already set!")
                walk_telo = telo_ref[node]['end']

        # If the longest walk starting from this node is already memoised and telomere is compatible, return it
        if node in memo: 
            memo_telo = memo[node][3]
            if walk_telo is None or memo_telo != walk_telo:
                return memo[node][0], memo[node][1], memo[node][2]

        visited.add(node)
        max_walk, max_key_nodes, min_penalty = [node], 0, 0

        # Traverse all the neighbors of the current node
        for neighbor in adj_list.get_neighbours(node):
            # Check visited
            dst = neighbor.new_dst_nid
            if dst in visited: continue
            # Check telomere compatibility
            terminate = False
            if walk_telo and dst < n_old_walks:
                curr_telo = get_telo_type(telo_ref, dst)
                if curr_telo:
                    if walk_telo != curr_telo:
                        terminate = True
                    else:
                        continue # ++/-- telomere connection

            if terminate:
                # Terminate search at the next node due to telomere compatibility
                current_walk, current_key_nodes, current_penalty = [dst], 1, 0
            else:
                # Perform DFS on the neighbor and check the longest walk from that neighbor
                current_walk, current_key_nodes, current_penalty = dfs(dst, visited, walk_telo)

            # Add the penalty for selecting that neighbour, either based on OL Len or OL Sim
            if penalty == "ol_len":
                if neighbor.ol_len < OL_LEN_CUTOFF_90: current_penalty += (OL_LEN_CUTOFF_90-neighbor.ol_len)
            elif penalty == "ol_sim":
                current_penalty += -1*neighbor.ol_sim

            if current_walk[-1] >= n_old_walks: # last node is a ghost node, should not count their penalty
                prev_node = node if len(current_walk) == 1 else current_walk[-2]
                curr_edge = adj_list.get_edge(prev_node, current_walk[-1])
                if penalty == "ol_len":
                    if curr_edge.ol_len < OL_LEN_CUTOFF_90: current_penalty -= (OL_LEN_CUTOFF_90-curr_edge.ol_len)
                elif penalty == "ol_sim":
                    current_penalty -= curr_edge.ol_sim

            # If adding this walk leads to a longer path, or same one with same length but lower penalty, update the max_walk and min_penalty
            if (current_key_nodes > max_key_nodes) or (current_key_nodes == max_key_nodes and current_penalty < min_penalty):
                max_walk = [node] + current_walk
                max_key_nodes = current_key_nodes
                min_penalty = current_penalty

        visited.remove(node)
        if node < n_old_walks: max_key_nodes += 1

        # Memoize the result for this node if chances are used up
        memo_counts[node] += 1
        if memo_counts[node] >= memo_chances:
            memo[node] = (max_walk, max_key_nodes, min_penalty, walk_telo)

        return max_walk, max_key_nodes, min_penalty    

    # Start DFS from the given start node
    res_walk, res_key_nodes, res_penalty = dfs(start_node, set(), None)

    # If the last node in a walk is a ghost node, remove it from the walk and negate its penalty.
    # This case should not occur, but I am just double checking
    if res_walk[-1] >= n_old_walks:
        curr_edge = adj_list.get_edge(res_walk[-2], res_walk[-1])
        if penalty == "ol_len":
            res_penalty -= curr_edge.ol_len
        elif penalty == "ol_sim":
            res_penalty -= curr_edge.ol_sim
        res_walk.pop()

    return res_walk, res_key_nodes, res_penalty

def get_walks(walk_ids, adj_list, telo_ref):
    # Generating new walks using greedy DFS
    new_walks = []
    temp_walk_ids, temp_adj_list = deepcopy(walk_ids), deepcopy(adj_list)
    n_old_walks = len(temp_walk_ids)

    # Remove all old walks that have both start and end telo regions
    for walk_id, v in telo_ref.items():
        if v['start'] and v['end']:
            new_walks.append([walk_id])
            temp_adj_list.remove_node(walk_id)
            temp_walk_ids.remove(walk_id)

    # Loop until all walks are connected
    while temp_walk_ids:
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in temp_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, hyperparams['dfs_penalty'])
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty):
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk
                best_penalty = curr_penalty

        for w in best_walk:
            temp_adj_list.remove_node(w)
            if w < n_old_walks: temp_walk_ids.remove(w)

        new_walks.append(best_walk)

    print(f"New walks generated! n new walks: {len(new_walks)}")
    return new_walks

def get_walks_telomere(walk_ids, adj_list, telo_ref):
    # Generating new walks using greedy DFS
    new_walks = []
    temp_walk_ids, temp_adj_list = deepcopy(walk_ids), deepcopy(adj_list)
    n_old_walks = len(temp_walk_ids)

    # Remove all old walks that have both start and end telo regions
    for walk_id, v in telo_ref.items():
        if v['start'] and v['end']:
            new_walks.append([walk_id])
            temp_adj_list.remove_node(walk_id)
            temp_walk_ids.remove(walk_id)

    # Split walks into those with telomeric regions and those without
    telo_walk_ids, non_telo_walk_ids = [], []
    for i in temp_walk_ids:
        if telo_ref[i]['start']:
            telo_walk_ids.append(i)
        else:
            non_telo_walk_ids.append(i)

    # Generate walks for walks with telomeric regions first
    while telo_walk_ids:
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in telo_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, hyperparams['dfs_penalty'])
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty):
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk
                best_penalty = curr_penalty

        for w in best_walk:
            temp_adj_list.remove_node(w)
            if w < n_old_walks: 
                if w in telo_walk_ids:
                    telo_walk_ids.remove(w)
                else:
                    non_telo_walk_ids.remove(w)

        new_walks.append(best_walk)

    assert len(telo_walk_ids) == 0, "Telomeric walks not all used!"

    # Generate walks for the rest
    while non_telo_walk_ids:
        best_walk, best_key_nodes, best_penalty = [], 0, 0
        for walk_id in non_telo_walk_ids: # the node_id is also the index
            curr_walk, curr_key_nodes, curr_penalty = get_best_walk(temp_adj_list, walk_id, n_old_walks, telo_ref, hyperparams['dfs_penalty'])
            if curr_key_nodes > best_key_nodes or (curr_key_nodes == best_key_nodes and curr_penalty < best_penalty):
                best_key_nodes = curr_key_nodes
                best_walk = curr_walk
                best_penalty = curr_penalty

        for w in best_walk:
            temp_adj_list.remove_node(w)
            if w < n_old_walks: non_telo_walk_ids.remove(w)

        new_walks.append(best_walk)

    print(f"New walks generated! n new walks: {len(new_walks)}")
    return new_walks

def get_telo_type(telo_ref, nid):
    x, y = telo_ref[nid]['start'], telo_ref[nid]['end']
    if x is not None:
        return x
    else:
        return y

# def get_telo_ref(old_walks, n2s, graph_path):
#     """
#     Generates telomere info for old walks.

#     telo_ref = {
#         walk_id_1 : {
#             'start' : '+', '-', or None,
#             'end' : '+', '-', or None
#         },
#         walk_id_2 : { ... },
#         ...
#     }
#     """

#     print("Generating old sequences...")
#     g = dgl.load_graphs(graph_path)[0][0]
#     edges = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This is from Martin's script)
#     for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
#         src, dst = src.item(), dst.item()
#         edges[(src, dst)] = idx

#     contigs = []
#     for i, walk in enumerate(old_walks):
#         prefixes = [(src, g.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]
#         res = []
#         for (src, prefix) in prefixes:
#             seq = str(n2s[src])
#             res.append(seq[:prefix])
#         contig = Seq.Seq(''.join(res) + str(n2s[walk[-1]]))  # TODO: why is this map here? Maybe I can remove it if I work with strings
#         contig = SeqIO.SeqRecord(contig)
#         contig.id = f'contig_{i+1}'
#         contig.description = f'length={len(contig)}'
#         contigs.append(contig)

#     print("Generating telomere info...")
#     def count_rep_region(seq, targ):
#         count, start = 0, 0
#         while True:
#             start = seq.find(targ, start)  # Find next occurrence
#             if start == -1:  # If no more occurrences are found
#                 break
#             count += 1  # Increment the count
#             start += len(targ)  # Move past this occurrence
#         return count
    
#     telo_ref = {}
#     for i, contig in enumerate(contigs):
#         curr_res = {'start':None, 'end':None}
#         seq = str(contig.seq)
#         len_seq = int(contig.description.split("=")[1])
        
#         for j in ['start', 'end']:
#             cutoff = int(min(MAX_SEARCH_REGION_LEN, len_seq//(1/SEARCH_REGION_P)))
#             c_seq = seq[:cutoff] if j == 'start' else seq[-cutoff:]
#             rep1_count, rep2_count = count_rep_region(c_seq, REP1), count_rep_region(c_seq, REP2)
#             if rep1_count > rep2_count:
#                 c_percentage = (rep1_count*6)/cutoff
#                 if c_percentage >= REP_THRESHOLD: curr_res[j] = '+'
#             else:
#                 c_percentage = (rep2_count*6)/cutoff
#                 if c_percentage >= REP_THRESHOLD: curr_res[j] = '-'

#         telo_ref[i] = curr_res

#     start_count, end_count, both_count = 0, 0, 0
#     for v in telo_ref.values():
#         if v['start'] and v['end']:
#             if v['start'] == v['end']: print("WARNING: Found same telomeric region in start and end of old walk!")
#             both_count += 1
#         if v['start']:
#             start_count += 1
#         if v['end']:
#             end_count += 1
#     print(f"Start telo walk count: {start_count}, End telo walk count: {end_count}, Both telo walk count: {both_count}")

#     return telo_ref

def get_contigs(old_walks, new_walks, adj_list, n2s, n2s_ghost, graph_path):
    g = dgl.load_graphs(graph_path)[0][0]
    n_old_walks = len(old_walks)

    print("Preprocessing walks...")
    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This is from Martin's script)
    for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

    walk_nodes, walk_seqs, walk_prefix_lens = [], [], []
    for i, walk in enumerate(new_walks):
        c_nodes, c_seqs, c_prefix_lens = [], [], []
        for j, node in enumerate(walk):
            if node >= n_old_walks: # Node is a new ghost node
                c_nodes.append(node)
                c_seqs.append(str(n2s_ghost[node]))
                curr_edge = adj_list.get_edge(node, walk[j+1])
                c_prefix_lens.append(curr_edge.prefix_len)
            else: # Node is an original walk
                old_walk = old_walks[node]
                if j == 0:
                    start = 0
                else:
                    curr_edge = adj_list.get_edge(walk[j-1], node)
                    start = old_walk.index(curr_edge.old_dst_nid)
                
                if j+1 == len(walk):
                    end = len(old_walk)-1
                    prefix_len = None
                else:
                    curr_edge = adj_list.get_edge(node, walk[j+1])
                    end = old_walk.index(curr_edge.old_src_nid)
                    prefix_len = curr_edge.prefix_len

                for k in range(start, end+1):
                    c_nodes.append(old_walk[k])
                    c_seqs.append(str(n2s[old_walk[k]]))

                    if k != end:
                        c_prefix_lens.append(g.edata['prefix_length'][edges_full[(old_walk[k], old_walk[k+1])]])

                if prefix_len: c_prefix_lens.append(prefix_len)

        walk_nodes.append(c_nodes)
        walk_seqs.append(c_seqs)
        walk_prefix_lens.append(c_prefix_lens)

    print(f"Generating sequences...")
    contigs = []
    for i, seqs in enumerate(walk_seqs):
        prefix_lens = walk_prefix_lens[i]
        c_contig = []
        for j, seq in enumerate(seqs[:-1]):
            c_contig.append(seq[:prefix_lens[j]])
        c_contig.append(seqs[-1])

        c_contig = Seq.Seq(''.join(c_contig))
        c_contig = SeqIO.SeqRecord(c_contig)
        c_contig.id = f'contig_{i+1}'
        c_contig.description = f'length={len(c_contig)}'
        contigs.append(c_contig)

    return contigs

def asm_metrics(contigs, save_path, ref_path):
    print(f"Saving assembly...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    asm_path = save_path+"0_assembly.fasta"
    SeqIO.write(contigs, asm_path, 'fasta')

    print(f"Running minigraph...")
    paf = save_path+"asm.paf"
    cmd = f'/home/stumanuel/GitHub/minigraph/minigraph -t32 -xasm -g10k -r10k --show-unmap=yes {ref_path} {asm_path}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()

    print(f"Running paftools...")
    paftools_path = "/home/stumanuel/GitHub/minimap2/misc/paftools.js"
    cmd = f'k8 {paftools_path} asmstat {ref_path+".fai"} {paf}'.split()
    report = save_path+"minigraph.txt"
    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()
    with open(report) as f:
        report = f.read()
        print(report)

def paf_postprocessing(name, hyperparams, paths):
    """
    Postprocesses walks from original pipeline to connect them with ghost data.
    *IMPORTANT
    - Only uses information from ghost-1 now. Any two walks are at most connected by a single ghost node. Also, all added ghost nodes must have at least one incoming and one outgoing edge to a walk. This is especially relevant in the section on Generating New Walks.
    """
    time_start = datetime.now()

    print(f"\n===== BEGIN FOR {name} =====")
    hyperparams_str = ""
    for k, v in hyperparams.items():
        hyperparams_str += f"{k}: {v}, "
    print(hyperparams_str[:-2]+"\n")

    print(f"Loading files... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    with open(paths['walks_path'], 'rb') as f:
        walks = pickle.load(f)
    with open(paths['fasta_path'], 'rb') as f:
        fasta_data = pickle.load(f)
    with open(paths['n2s_path'], 'rb') as f:
        n2s = pickle.load(f)
    with open(paths['paf_path'], 'rb') as f:
        paf_data = pickle.load(f)

    print(f"Chopping old walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    chop_walk_min_count = hyperparams['chop_walk_min_count_ref'][name] if name in hyperparams['chop_walk_min_count_ref'] else CHOP_WALK_MIN_COUNT_DEFAULT
    walks, telo_ref = chop_walks(walks, n2s, chop_walk_min_count)

    n_id = 0
    adj_list = AdjList()

    # Only the first and last walk_valid_p% of nodes in a walk can be connected. Also initialises nodes from walks
    n2n_start, n2n_end = {}, {} # n2n maps old n_id to new n_id, for the start and ends of the walks respectively
    walk_ids = [] # all n_ids that belong to walks
    for walk in walks:
        if len(walk) == 1:
            n2n_start[walk[0]] = n_id
            n2n_end[walk[0]] = n_id
        else:
            cutoff = int(max(1, len(walk) // (1/hyperparams['walk_valid_p'])))
            first_part, last_part = walk[:cutoff], walk[-cutoff:]
            for n in first_part:
                n2n_start[n] = n_id
            for n in last_part:
                n2n_end[n] = n_id

        walk_ids.append(n_id)
        n_id += 1

    n_old_walks = len(walk_ids)

    print(f"Adding edges between existing nodes... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    ol_len_cutoff = 0
    if hyperparams['ol_len_cutoff'] == 50:
        ol_len_cutoff = OL_LEN_CUTOFF_50
    elif hyperparams['ol_len_cutoff'] == 75:
        ol_len_cutoff = OL_LEN_CUTOFF_75
    elif hyperparams['ol_len_cutoff'] == 90:
        ol_len_cutoff = OL_LEN_CUTOFF_90

    valid_src, valid_dst, prefix_lens, ol_lens, ol_sims, ghost_data = paf_data['valid_src'], paf_data['valid_dst'], paf_data['prefix_len'], paf_data['ol_len'], paf_data['ol_similarity'], paf_data['ghost_data']
    added_edges_count = 0
    for i in range(len(valid_src)):
        src, dst, prefix_len, ol_len, ol_sim = valid_src[i], valid_dst[i], prefix_lens[i], ol_lens[i], ol_sims[i]
        if ol_len < ol_len_cutoff: continue
        if src in n2n_end and dst in n2n_start:
            if n2n_end[src] == n2n_start[dst]: continue # ignore self-edges
            added_edges_count += 1
            adj_list.add_edge(Edge(
                new_src_nid=n2n_end[src], 
                new_dst_nid=n2n_start[dst], 
                old_src_nid=src, 
                old_dst_nid=dst, 
                prefix_len=prefix_len, 
                ol_len=ol_len, 
                ol_sim=ol_sim
            ))
    print("Added edges:", added_edges_count)

    with open(paths['r2n_path'], 'rb') as f:
        r2n = pickle.load(f)

    print(f"Adding ghost nodes and edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    n2s_ghost = {}
    ghost_data = ghost_data['hop_1'] # WE ONLY DO FOR 1-HOP FOR NOW
    added_nodes_count = 0
    for orient in ['+', '-']:
        for read_id, data in ghost_data[orient].items():
            curr_out_neighbours, curr_in_neighbours = set(), set()

            for i, out_read_id in enumerate(data['outs']):
                out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
                if out_n_id not in n2n_start: continue
                if data['ol_len_outs'][i] < ol_len_cutoff: continue
                curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_similarity_outs'][i]))

            for i, in_read_id in enumerate(data['ins']):
                in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
                if in_n_id not in n2n_end: continue
                if data['ol_len_ins'][i] < ol_len_cutoff: continue
                curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_similarity_ins'][i]))

            # ghost nodes are only useful if they have both at least one outgoing and one incoming edge
            if not curr_out_neighbours or not curr_in_neighbours: continue

            for n in curr_out_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n_id,
                    new_dst_nid=n2n_start[n[0]],
                    old_src_nid=None,
                    old_dst_nid=n[0],
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))
            for n in curr_in_neighbours:
                adj_list.add_edge(Edge(
                    new_src_nid=n2n_end[n[0]],
                    new_dst_nid=n_id,
                    old_src_nid=n[0],
                    old_dst_nid=None,
                    prefix_len=n[1],
                    ol_len=n[2],
                    ol_sim=n[3]
                ))

            seq = fasta_data[read_id][0] if orient == '+' else fasta_data[read_id][1]
            n2s_ghost[n_id] = seq
            n_id += 1
            added_nodes_count += 1
    print("Final number of nodes:", n_id)

    if not added_edges_count and not added_nodes_count:
        print("No suitable nodes and edges found to add to these walks. Returning...")
        return

    # Remove duplicate edges between nodes. If there are multiple connections between a walk and another node/walk, we choose the best one.
    # This could probably have been done while adding the edges in. However, to avoid confusion, i'm doing this separately.
    print(f"Removing duplicate edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    for new_src_nid, connected in adj_list.adj_list.items():
        dup_checker = {}
        for neigh in connected:
            new_dst_nid = neigh.new_dst_nid
            if new_dst_nid not in dup_checker:
                dup_checker[new_dst_nid] = neigh
            else:
                # duplicate is found
                og = dup_checker[new_dst_nid]
                if new_src_nid < n_old_walks and new_dst_nid < n_old_walks: # both are walks
                    walk_src, walk_dst = walks[new_src_nid], walks[new_dst_nid]
                    start_counting = None
                    score = 0
                    for i in reversed(walk_src):
                        if i == og.old_src_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '+'
                        elif i == neigh.old_src_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '-'

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    start_counting = None
                    for i in walk_dst:
                        if i == og.old_dst_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '+'
                        elif i == neigh.old_dst_nid:
                            if start_counting: break # both old and new have been found, and score updated
                            start_counting = '-'

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    if score < 0: # if score is < 0, new is better
                        dup_checker[new_dst_nid] = neigh
                elif new_src_nid < n_old_walks:
                    walk = walks[new_src_nid]
                    for i in reversed(walk):
                        if i == neigh.old_src_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                elif new_dst_nid < n_old_walks:
                    walk = walks[new_dst_nid]
                    for i in walk:
                        if i == neigh.old_dst_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[new_dst_nid] = neigh
                            break
                else:
                    raise ValueError("Duplicate edge between two non-walks found!")
                
        adj_list.adj_list[new_src_nid] = set(n for n in dup_checker.values())
    print("Final number of edges:", sum(len(x) for x in adj_list.adj_list.values()))

    # print(f"Generating telomere reference info... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    # telo_ref = get_telo_ref(walks, n2s, paths['graph_path'])

    print(f"Generating new walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    if hyperparams['walk_var'] == 'default':
        new_walks = get_walks(walk_ids, adj_list, telo_ref)
    elif hyperparams['walk_var'] == 'telomere':
        new_walks = get_walks_telomere(walk_ids, adj_list, telo_ref)
    else:
        raise ValueError("Invalid walk_var!")

    ### FOR DEBUGGING ###
    # print(adj_list)
    # print(new_walks)

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    contigs = get_contigs(walks, new_walks, adj_list, n2s, n2s_ghost, paths['graph_path'])

    print(f"Calculating assembly metrics... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    asm_metrics(contigs, paths['save_path'], paths['ref_path'])

    print(f"Run finished! (Time: {timedelta_to_str(datetime.now() - time_start)})")
    return

def run_paf_postprocessing(names, dataset, hyperparams):
    ref = {}
    for i in [1,3,5,9,12,18]:
        ref[i] = [i for i in range(15)]
    for i in [11,16,17,19,20]:
        ref[i] = [i for i in range(5)]

    if dataset=="haploid_train":
        for chr in names:
            for i in ref[chr]:
                name = f"chr{chr}_M_{i}"
                paths = {
                    'walks_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/train/{name}/walks.pkl",
                    'fasta_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_fasta_data.pkl",
                    'paf_path':f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl',
                    'n2s_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl",
                    'r2n_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_r2n.pkl",
                    'save_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/train/{name}/",
                    'ref_path':f"/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/centromeres/chr{chr}_MATERNAL_centromere.fasta",
                    'graph_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl"
                }
                paf_postprocessing(name=name, hyperparams=hyperparams, paths=paths)
    elif dataset=="haploid_test":
        test_ref = {
            'chm13' : '/mnt/sod2-project/csb4/wgs/martin/genome_references/chm13_v11/chm13_full_v1_1.fasta',
            'arab' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/arabidopsis/latest/GWHBDNP00000000.1.genome.fasta',
            'mouse' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/mus_musculus/mmusculus_GRCm39.fna',
            'chicken' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/bGalGal1/maternal/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna'
        }
        for name in names:
            paths={
                'walks_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl",
                'fasta_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_fasta_data.pkl",
                'paf_path':f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl',
                'n2s_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl",
                'r2n_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_r2n.pkl",
                'save_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/{name}/",
                'ref_path':test_ref[name],
                'graph_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl"                
            }
            paf_postprocessing(name=name, hyperparams=hyperparams, paths=paths)
    elif dataset=="diploid":
        for name in names:
            name1 = f"hg002_v101_chr{name}_0"
            for v in ['m', 'p']:
                name2 = f"chr_{name}_synth_{v}"
                name3 = f"chr{name}_MATERNAL.fasta" if v == 'm' else f"chr{name}_PATERNAL.fasta"
                paths = {
                    'walks_path':f"/mnt/sod2-project/csb4/wgs/martin/assemblies/{name2}/walks.pkl",
                    'fasta_path':f"/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset/full_reads/{name1}.pkl",
                    'paf_path':f'/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset/paf/{name1}.pkl',
                    'n2s_path':f"/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset/reduced_reads/{name1}.pkl", 
                    'r2n_path':f"/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset/read_to_node/{name1}.pkl",
                    'save_path':f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/{name2}/",
                    'ref_path':f"/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/chromosomes/{name3}",
                    'graph_path':f"/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset/dgl_graphs/{name1}.dgl"                    
                }
                paf_postprocessing(name=name2, hyperparams=hyperparams, paths=paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='haploid_train', help="haploid_train, haploid_test, or diploid")
    parser.add_argument("--walk_valid_p", type=float, default=0.02)
    parser.add_argument("--ol_len_cutoff", type=int, default=0, help="50, 75, or 90")
    parser.add_argument("--dfs_penalty", type=str, default=None, help="ol_len or ol_sim, leave blank for no penalty")
    parser.add_argument("--walk_var", type=str, default='default', help="default or telomere")
    args = parser.parse_args()
    dataset = args.dataset
    hyperparams = {
        'chop_walk_min_count_ref' : {
            'arab' : 100,
            'chicken' : 1650,
            'mouse' : 500,
            'chm13' : 200
        },
        'walk_valid_p' : args.walk_valid_p,
        'ol_len_cutoff' : args.ol_len_cutoff,
        'dfs_penalty' : args.dfs_penalty,
        'walk_var' : args.walk_var
    }

    if dataset == "haploid_train":
        names = [1,3,5,9,11,12,16,17,18,19,20]
    elif dataset == "haploid_test":
        names = ["arab", "chicken", "mouse", "chm13"]
    elif dataset == "diploid":
        names = [10,1,5,18,19]

    # run_paf_postprocessing(names, dataset=dataset, hyperparams=hyperparams)

    for names in [["chicken"], ["arab"], ["chm13"], ["mouse"]]:
        for walk_var in ["default", "telomere"]:
            hyperparams["walk_var"] = walk_var
            run_paf_postprocessing(names, dataset="haploid_test", hyperparams=hyperparams)
