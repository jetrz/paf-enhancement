import dgl, os, pickle, subprocess
from Bio import Seq, SeqIO
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'

def find_in_adj_list(neighbours, val, return_src):
    for i in neighbours:
        if i[0] == val:
            if return_src:
                return i[1], i[3]
            else:
                return i[2], i[3]
            
def longest_walk(adj_list, start_node):
    # Dictionary to memoize the longest walk from each node
    memo = {}

    def dfs(node, visited):
        # If the longest walk starting from this node is already computed, return it
        if node in memo:
            return memo[node]
        visited.add(node)

        # Initialize the maximum walk from this node to just the node itself
        max_walk = [node]

        # Traverse all the neighbors of the current node
        for neighbor in adj_list.get(node, []):
            neighbor = neighbor[0]
            if neighbor in visited: continue
            # Perform DFS on the neighbor and check the longest walk from that neighbor
            current_walk = dfs(neighbor, visited)

            # If adding this walk leads to a longer path, update the max_walk
            if len(current_walk) + 1 > len(max_walk):
                max_walk = [node] + current_walk

        visited.remove(node)

        # Memoize the result for this node
        memo[node] = max_walk
        return max_walk

    # Start DFS from the given start node
    return dfs(start_node, set())

def paf_decoding(name, walk_valid_p, walks_path, fasta_path, paf_path, n2s_path, r2n_path, save_path, ref_path, graph_path):
    """
    Postprocesses walks from original pipeline to connect them with ghost data.
    *IMPORTANT
    - Only uses information from ghost-1 now. Any two walks are at most connected by a single ghost node. Also, all added ghost nodes must have at least one incoming and one outgoing edge to a walk. This is especially relevant in the section on Generating New Walks.
    """
    time_start = datetime.now()

    print(f"\n===== BEGIN FOR {name} =====\n")
    print(f"Loading files... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    with open(walks_path, 'rb') as f:
        walks = pickle.load(f)
    with open(fasta_path, 'rb') as f:
        fasta_data = pickle.load(f)
    with open(n2s_path, 'rb') as f:
        n2s = pickle.load(f)

    n_id = 0
    adj_list = {}

    # Only the first and last walk_valid_p% of nodes in a walk can be connected. Also initialises nodes from walks
    n2n_start, n2n_end = {}, {} # n2n maps old n_id to new n_id, for the start and ends of the walks respectively
    walk_ids = [] # all n_ids that belong to walks
    for walk in walks:
        if len(walk) == 1:
            n2n_start[walk[0]] = n_id
            n2n_end[walk[0]] = n_id
        else:
            cutoff = int(max(1, len(walk) // (1/walk_valid_p)))
            first_part, last_part = walk[:cutoff], walk[-cutoff:]
            for n in first_part:
                n2n_start[n] = n_id
            for n in last_part:
                n2n_end[n] = n_id

        adj_list[n_id] = set() # adjacency list has the format: new src n_id : [(new dst n_id, old src n_id, old dst n_id), ... ]
        walk_ids.append(n_id)

        n_id += 1

    n_old_walks = len(walk_ids)
    print(f"Number of original walks: {n_old_walks}")

    with open(paf_path, 'rb') as f:
        paf_data = pickle.load(f)
        valid_src, valid_dst, prefix_lens, ghost_data = paf_data['valid_src'], paf_data['valid_dst'], paf_data['prefix_len'], paf_data['ghost_data']

    print(f"Adding edges between existing nodes... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    added_edges_count = 0
    for i in range(len(valid_src)):
        src, dst, prefix_len = valid_src[i], valid_dst[i], prefix_lens[i]
        if src in n2n_end and dst in n2n_start:
            if n2n_end[src] == n2n_start[dst]: continue # ignore self-edges
            added_edges_count += 1
            adj_list[n2n_end[src]].add((n2n_start[dst], src, dst, prefix_len))
    print("Added edges:", added_edges_count)

    with open(r2n_path, 'rb') as f:
        r2n = pickle.load(f)

    print(f"Adding ghost nodes and edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    ghost_data = ghost_data['hop_1'] # WE ONLY DO FOR 1-HOP FOR NOW
    added_nodes_count = 0
    for read_id, data in ghost_data['+'].items():
        curr_out_neighbours, curr_in_neighbours = set(), set()

        for i, out_read_id in enumerate(data['outs']):
            out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
            if out_n_id not in n2n_start: continue
            curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i]))

        for i, in_read_id in enumerate(data['ins']):
            in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
            if in_n_id not in n2n_end: continue
            curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i]))

        # ghost nodes are only useful if they have both at least one outgoing and one incoming edge
        if not curr_out_neighbours or not curr_in_neighbours: continue

        adj_list[n_id] = set()
        for n in curr_out_neighbours:
            adj_list[n_id].add((n2n_start[n[0]], None, n[0], n[1]))
        for n in curr_in_neighbours:
            adj_list[n2n_end[n[0]]].add((n_id, n[0], None, n[1]))

        seq = fasta_data[read_id][0]
        n2s[n_id] = seq
        n_id += 1
        added_nodes_count += 1

    for read_id, data in ghost_data['-'].items():
        curr_out_neighbours, curr_in_neighbours = set(), set()
        
        for i, out_read_id in enumerate(data['outs']):
            out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
            if out_n_id not in n2n_start: continue
            curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i]))

        for i, in_read_id in enumerate(data['ins']):
            in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
            if in_n_id not in n2n_end: continue
            curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i]))

        # ghost nodes are only useful if they have both at least one outgoing and one incoming edge
        if not curr_out_neighbours or not curr_in_neighbours: continue

        adj_list[n_id] = set()
        for n in curr_out_neighbours:
            adj_list[n_id].add((n2n_start[n[0]], None, n[0], n[1]))
        for n in curr_in_neighbours:
            adj_list[n2n_end[n[0]]].add((n_id, n[0], None, n[1]))

        seq = fasta_data[read_id][1]
        n2s[n_id] = seq
        n_id += 1
        added_nodes_count += 1
    print("Final number of nodes:", n_id)

    if not added_edges_count and not added_nodes_count:
        print("No suitable nodes and edges found to add to these walks. Returning...")
        return

    # Remove duplicate edges between nodes. If there are multiple connections between a node and another, at least one of them must be a walk. 
    # In that case, we choose the best one.
    # This could probably have been done while adding the edges in. However, to avoid confusion, i'm doing this separately.
    print(f"Removing duplicate edges... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    dup_checker = {}
    for new_src_nid, connected in deepcopy(adj_list).items():
        for i in connected:
            new_dst_nid, old_src_nid, old_dst_nid, prefix_len = i[0], i[1], i[2], i[3]
            if (new_src_nid, new_dst_nid) not in dup_checker:
                dup_checker[(new_src_nid, new_dst_nid)] = (old_src_nid, old_dst_nid, prefix_len)
            else:
                # duplicate is found
                og = dup_checker[(new_src_nid, new_dst_nid)]
                if new_src_nid < n_old_walks and new_dst_nid < n_old_walks: # both are walks
                    walk_src, walk_dst = walks[new_src_nid], walks[new_dst_nid]
                    start_counting = None
                    score = 0
                    for i in reversed(walk_src):
                        if i == og[0]:
                            start_counting = '+'
                            if start_counting: break # both old and new have been found, and score updated
                        elif i == old_src_nid:
                            start_counting = '-'
                            if start_counting: break # both old and new have been found, and score updated

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    start_counting = None
                    for i in walk_dst:
                        if i == og[1]:
                            start_counting = '+'
                            if start_counting: break # both old and new have been found, and score updated
                        elif i == old_dst_nid:
                            start_counting = '-'
                            if start_counting: break # both old and new have been found, and score updated

                        if start_counting == '+':
                            score += 1
                        elif start_counting == '-':
                            score -= 1

                    if score < 0: # if score is < 0, new is better, change and remove old one from adj list
                        dup_checker[(new_src_nid, new_dst_nid)] = (old_src_nid, old_dst_nid, prefix_len)
                        adj_list[new_src_nid].remove((new_dst_nid, og[0], og[1], og[2]))
                    else: # remove new one from adj list
                        adj_list[new_src_nid].remove((new_dst_nid, old_src_nid, old_dst_nid, prefix_len))
                elif new_src_nid < n_old_walks:
                    walk = walks[new_src_nid]
                    for i in reversed(walk):
                        if i == og[0]: # old one is better, remove the new one from adj_list
                            adj_list[new_src_nid].remove((new_dst_nid, old_src_nid, old_dst_nid, prefix_len))
                            break
                        elif i == old_src_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[(new_src_nid, new_dst_nid)] = (old_src_nid, old_dst_nid, prefix_len)
                            adj_list[new_src_nid].remove((new_dst_nid, og[0], og[1], og[2]))
                            break
                elif new_dst_nid < n_old_walks:
                    walk = walks[new_dst_nid]
                    for i in walk:
                        if i == og[1]: # old one is better, remove the new one from adj_list
                            adj_list[new_src_nid].remove((new_dst_nid, old_src_nid, old_dst_nid, prefix_len))
                            break
                        elif i == old_dst_nid: # new one is better, update dupchecker and remove old one from adj list
                            dup_checker[(new_src_nid, new_dst_nid)] = (old_src_nid, old_dst_nid, prefix_len)
                            adj_list[new_src_nid].remove((new_dst_nid, og[0], og[1], og[2]))
                            break
                else:
                    raise ValueError("Duplicate edge between two non-walks found!")
    print("Final number of edges:", sum(len(x) for x in adj_list.values()))

    # Generating new walks using greedy DFS
    print(f"Generating new walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    new_walks = []
    temp_walk_ids = deepcopy(walk_ids)
    temp_adj_list = deepcopy(adj_list)
    while temp_walk_ids: # Loop until all walks are connected
        best_walk, best_walk_count = [], 0
        for walk_id in temp_walk_ids: # the node_id is also the index
            if walk_id not in temp_adj_list or not temp_adj_list[walk_id]:
                max_walk = [walk_id]
            else:
                max_walk = longest_walk(temp_adj_list, walk_id)
                if max_walk[-1] >= n_old_walks: max_walk.pop() # Remove last item in walk if it is not a key node

            n_key_nodes = (len(max_walk)+1)/2
            if n_key_nodes > best_walk_count:
                best_walk_count = n_key_nodes
                best_walk = max_walk

        for w in best_walk:
            if w in temp_adj_list: del temp_adj_list[w]
            for k, v in deepcopy(temp_adj_list).items():
                curr_n = [x for x in v if x[0] != w]
                if curr_n:
                    temp_adj_list[k] = curr_n
                else:
                    del temp_adj_list[k]

            if w < n_old_walks: temp_walk_ids.remove(w)

        new_walks.append(best_walk)
    print(f"New walks generated! n new walks: {len(new_walks)}")

    print(f"Preprocessing walks... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    g = dgl.load_graphs(graph_path)[0][0]
    # Create a list of all edges
    edges_full = {}  ## I dont know why this is necessary. but when cut transitives some eges are wrong otherwise.
    for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

    walk_nodes, walk_seqs, walk_prefix_lens = [], [], []
    for i, walk in enumerate(new_walks):
        c_nodes, c_seqs, c_prefix_lens = [], [], []
        for j, node in enumerate(walk):
            if node >= n_old_walks: # Node is a new ghost node
                c_nodes.append(node)
                c_seqs.append(str(n2s[node]))
                _, prefix_len = find_in_adj_list(adj_list[node], walk[j+1], return_src=True)
                c_prefix_lens.append(prefix_len)
            else: # Node is an original walk
                old_walk = walks[node]
                if j == 0:
                    start = 0
                else:
                    dst, _ = find_in_adj_list(adj_list[walk[j-1]], node, return_src=False)
                    start = old_walk.index(dst)
                
                if j+1 == len(walk):
                    end = len(old_walk)-1
                    prefix_len = None
                else:
                    src, prefix_len = find_in_adj_list(adj_list[node], walk[j+1], return_src=True)
                    end = old_walk.index(src)

                for k in range(start, end+1):
                    c_nodes.append(old_walk[k])
                    c_seqs.append(str(n2s[old_walk[k]]))

                    if k != end:
                        c_prefix_lens.append(g.edata['prefix_length'][edges_full[(old_walk[k], old_walk[k+1])]])

                if prefix_len: c_prefix_lens.append(prefix_len)

        # print("walk complete. n nodes:", len(c_nodes), "n_seqs:", len(c_seqs), "n_prefix_lens:", len(c_prefix_lens))
        walk_nodes.append(c_nodes)
        walk_seqs.append(c_seqs)
        walk_prefix_lens.append(c_prefix_lens)

    print(f"Generating contigs... (Time: {timedelta_to_str(datetime.now() - time_start)})")
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

    print(f"Saving assembly... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    asm_path = save_path+"0_assembly.fasta"
    SeqIO.write(contigs, asm_path, 'fasta')

    print(f"Running minigraph... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    paf = save_path+"asm.paf"
    cmd = f'/home/stumanuel/GitHub/minigraph/minigraph -t32 -xasm -g10k -r10k --show-unmap=yes {ref_path} {asm_path}'.split(' ')
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()

    print(f"Running paftools... (Time: {timedelta_to_str(datetime.now() - time_start)})")
    paftools_path = "/home/stumanuel/GitHub/minimap2/misc/paftools.js"
    cmd = f'k8 {paftools_path} asmstat {ref_path+".fai"} {paf}'.split()
    report = save_path+"minigraph.txt"
    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()
    with open(report) as f:
        report = f.read()
        print(report)

def run_paf_decoding(names, walk_valid_p=0.25, train=True):
    ref = {}
    for i in [1,3,5,9,12,18]:
        ref[i] = [i for i in range(15)]
    for i in [11,16,17,19,20]:
        ref[i] = [i for i in range(5)]

    if train:
        for chr in names:
            for i in ref[chr]:
                name = f"chr{chr}_M_{i}"
                paf_decoding(
                    name=name, 
                    walk_valid_p=walk_valid_p,
                    walks_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/train/{name}/walks.pkl",
                    fasta_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_fasta_data.pkl",
                    paf_path=f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl',
                    n2s_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl",
                    r2n_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_r2n.pkl",
                    save_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/train/{name}/",
                    ref_path=f"/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/centromeres/chr{chr}_MATERNAL_centromere.fasta",
                    graph_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl"
                )
    else:
        test_ref = {
            'chm13' : '/mnt/sod2-project/csb4/wgs/martin/genome_references/chm13_v11/chm13_full_v1_1.fasta',
            'arab' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/arabidopsis/latest/GWHBDNP00000000.1.genome.fasta',
            'mouse' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/mus_musculus/mmusculus_GRCm39.fna',
            'chicken' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/bGalGal1/maternal/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna'
        }
        for name in names:
            paf_decoding(
                name=name,
                walk_valid_p=walk_valid_p,
                walks_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl",
                fasta_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_fasta_data.pkl",
                paf_path=f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_paf_data.pkl',
                n2s_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl",
                r2n_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_r2n.pkl",
                save_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/{name}/",
                ref_path=test_ref[name],
                graph_path=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl"
            )

def analyse(chrs):
    ref = {}
    for i in [1,3,5,9,12,18]:
        ref[i] = [i for i in range(15)]
    for i in [11,16,17,19,20]:
        ref[i] = [i for i in range(5)]

    path= "/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/postprocessed/train/"
    for chr in chrs:
        l_cov, ng50, nga50, rdup = [], [], [], []
        for i in ref[chr]:
            name = f"chr{chr}_M_{i}"
            if os.path.isdir(path+name):
                with open(path+name+"/minigraph.txt") as f:
                    report = f.readlines()
                for row in report:
                    if row.startswith('l_cov'):
                        l_cov.append(row.split('\t')[1].strip("\n"))
                    elif row.startswith('NG50'):
                        ng50.append(row.split('\t')[1].strip("\n"))
                    elif row.startswith('NGA50'):
                        nga50.append(row.split('\t')[1].strip("\n"))
                    elif row.startswith('Rdup'):
                        rdup.append(row.split('\t')[1].strip("\n"))
            else:
                l_cov.append(None); ng50.append(None); nga50.append(None); rdup.append(None)
        
        print(f"chr: {chr}\n lengths: {l_cov}\n ng50s: {ng50}\n nga50s: {nga50}\n rdups: {rdup}") 

        
if __name__ == "__main__":
    chrs=[1,3,5,9,11,12,16,17,18,19,20]
    names = ["chicken", "arab", "chm13"]
    run_paf_decoding(names, walk_valid_p=0.05, train=False)
    analyse(chrs)
            


    
