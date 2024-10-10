import dgl, os, pickle, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Seq, SeqIO
from tqdm import tqdm

from decoding_paf import AdjList, Edge

def analyse(name, hop):
    print(f"\n=== ANALYSING FOR {name} ===\n")
    g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/{name}.dgl")[0][0]
    # g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/chr1_M_0.dgl")[0][0]
    print("graph loaded! number of edges:", len(g.edata['E_ID']))

    # Extract edge data from the graph
    edge_data = g.edata

    # for i in ['node_hop', 'read_start', 'read_end', 'read_chr', 'read_strand', 'read_length']:
    #     print(f"{i}:", g.ndata[i][:20], g.ndata[i][20:])
    # print("\n")
    # for i in ['edge_hop', 'decision_edge_gt_only_neg', 'decision_edge_gt_only_pos', 'gt_17c', 'gt_soft', 'y']:
    #     print(f"{i}:", g.edata[i][:20], g.edata[i][20:])

    # Filter edges where edge_hop == 1
    edge_hop_mask = edge_data['edge_hop'] == hop
    gt_bin_0_mask = edge_data['gt_bin'] == 0
    combined_mask = edge_hop_mask & gt_bin_0_mask

    # Compute OL percentiles (50th, 75th, and 90th)
    filtered_overlap_length_gt_bin_0 = edge_data['overlap_length'][combined_mask].numpy()
    percentiles = [50, 75, 90]
    percentile_values = np.percentile(filtered_overlap_length_gt_bin_0, percentiles)
    print(f"OL 50th percentile: {percentile_values[0]}")
    print(f"OL 75th percentile: {percentile_values[1]}")
    print(f"OL 90th percentile: {percentile_values[2]}")

    # Get filtered features
    filtered_gt_bin = edge_data['gt_bin'][edge_hop_mask].numpy()
    filtered_overlap_length = edge_data['overlap_length'][edge_hop_mask].numpy()
    filtered_overlap_similarity = edge_data['overlap_similarity'][edge_hop_mask].numpy()

    # Create a DataFrame for correlation analysis
    df = pd.DataFrame({
        'gt_bin': filtered_gt_bin,
        'overlap_length': filtered_overlap_length,
        'overlap_similarity': filtered_overlap_similarity
    })

    # Plot the correlation using seaborn
    plt.figure(figsize=(10, 5))

    # Plot for overlap_length vs gt_bin
    plt.subplot(1, 2, 1)
    plt.hexbin(df['overlap_length'], df['gt_bin'], gridsize=30, cmap='Blues', mincnt=1)
    plt.colorbar(label='Point density')
    for perc, value in zip(percentiles, percentile_values):
        plt.axvline(x=value, linestyle='--', label=f'{perc}th Percentile: {value:.2f}')

    # Plot for overlap_similarity vs gt_bin
    plt.subplot(1, 2, 2)
    plt.hexbin(df['overlap_similarity'], df['gt_bin'], gridsize=30, cmap='Blues', mincnt=1)
    plt.colorbar(label='Point density')

    # Display the plot
    plt.tight_layout()
    # plt.savefig(f"graphs/train/{name}_{hop}_ol_corr.png")
    plt.clf()

    return percentile_values

# ref = {}
# for i in [1,3,5,9,12,18]:
#     ref[i] = [j for j in range(15)]
# for i in [11,16,17,19,20]:
#     ref[i] = [j for j in range(5)]

# res50, res75, res90 = [], [], []
# for chr in ref.keys():
#     selected = random.sample(ref[chr], 5)
#     for i in selected:
#         percentile_values = analyse(name=f"chr{chr}_M_{i}", hop=0)
#         res50.append(percentile_values[0]); res75.append(percentile_values[1]); res90.append(percentile_values[2])

# print(f"Final Average 50th Percentile Value: {sum(res50)/len(res50)}")
# print(f"Final Average 75th Percentile Value: {sum(res75)/len(res75)}")
# print(f"Final Average 90th Percentile Value: {sum(res90)/len(res90)}")

def analyse_reports(chrs):
    """
    Only for haploid_train.
    """
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

def count_seq(seq, targ):
    count, start = 0, 0
    while True:
        start = seq.find(targ, start)  # Find next occurrence
        if start == -1:  # If no more occurrences are found
            break
        count += 1  # Increment the count
        start += len(targ)  # Move past this occurrence

    return count

def analyse_telomere(name):
    print(f"\n=== ANALYSING TELOMERE INFO FOR {name} ===")
    rep1, rep2 = 'TTAGGG', 'CCCTAA'

    fasta_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/0_assembly.fasta"
    with open(fasta_path, 'rt') as handle:
        rows = SeqIO.parse(handle, 'fasta')
    
        res = {}
        for i, record in enumerate(tqdm(rows, ncols=120)):
            curr_res = {'start':{'len':None, 'n':None, 'type':None}, 'end':{'len':None, 'n':None, 'type':None}}
            seq = str(record.seq)
            len_seq = len(seq)
            
            for j in ['start', 'end']:
                cutoff = min(30000, len_seq//10)
                c_seq = seq[:cutoff] if j == 'start' else seq[-cutoff:]
                curr_res[j]['len'] = len(c_seq)
                rep1_count, rep2_count = count_seq(c_seq, rep1), count_seq(c_seq, rep2)
                print(f"i: {i}, j: {j}, cutoff: {cutoff}, rep1_count: {rep1_count}, rep2_count: {rep2_count}")
                if rep1_count > rep2_count:
                    curr_res[j]['n'] = rep1_count
                    curr_res[j]['type'] = '+'
                else:
                    curr_res[j]['n'] = rep2_count
                    curr_res[j]['type'] = '-'

            res[i] = curr_res

    percentages, percentages_rev, reals, reals_rev = [], [], [], []
    for k, v in res.items():
        for j in ['start', 'end']:
            curr = v[j]
            curr_percentage = (curr['n']*6)/curr['len']
            if curr['type'] == '+':
                percentages.append(curr_percentage)
                reals.append(curr['n'])
            else:
                percentages_rev.append(curr_percentage)
                reals_rev.append(curr['n'])

    percentages.sort(); percentages_rev.sort(); reals.sort(); reals_rev.sort()

    def plot(sorted_array, ax, color, label):
        # Calculate percentiles
        p50 = np.percentile(sorted_array, 50)
        p75 = np.percentile(sorted_array, 75)
        p90 = np.percentile(sorted_array, 90)
        
        # Plot the sorted array
        ax.plot(sorted_array, label=label, color=color)
        
        # Plot the percentiles
        ax.axhline(p50, color=color, linestyle='--', label='50th Percentile')
        ax.axhline(p75, color=color, linestyle=':', label='75th Percentile')
        ax.axhline(p90, color=color, linestyle='-', label='90th Percentile')
        
        # Set labels and title
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    plot(percentages, axs[0], 'blue', '+')
    plot(percentages_rev, axs[0], 'green', '-')
    plot(reals, axs[1], 'blue', '+')
    plot(reals_rev, axs[1], 'green', '-')
    plt.tight_layout()
    plt.savefig(f'graphs/telomere/{name}.png')
    plt.clf()

# analyse_telomere('arab')

def analyse_telomere_2(name, n, min_count_threshold):
    print(f"\n=== ANALYSING TELOMERE INFO FOR {name} ===")
    rep1, rep2 = 'TTAGGG', 'CCCTAA'

    def count_telo_rep(seq, n):
        i, rep1_count, rep2_count, middle_telo_count = 0, 0, 0, 0
        counts = []
        while i < len(seq):
            curr_rep = None
            if seq.startswith(rep1, i) or seq.startswith(rep2, i):
                if seq.startswith(rep1, i):
                    curr_rep = rep1
                elif seq.startswith(rep2, i):
                    curr_rep = rep2

                curr_count = 1
                end_index = i+6
                while end_index < len(seq):
                    next_index = seq.find(curr_rep, end_index)
                    if next_index == -1: break
                    if next_index - end_index <= n:
                        curr_count += 1
                        end_index = next_index + 6
                    else:
                        break

                i = end_index
                counts.append(curr_count)

                if curr_count >= min_count_threshold:
                    if curr_rep == rep1:
                        rep1_count += 1
                    else:
                        rep2_count += 1

                    if (i-curr_count*6) > n and (len(seq)-i) > n:
                        middle_telo_count += 1
            else:
                i += 1

        return rep1_count, rep2_count, counts, middle_telo_count

    fasta_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/0_assembly.fasta"
    with open(fasta_path, 'rt') as handle:
        rows = SeqIO.parse(handle, 'fasta')
    
        rep1_counts, rep2_counts, counts, middle_telo_counts = [], [], [], []
        for i, record in enumerate(tqdm(rows, ncols=120)):
            seq = str(record.seq)

            c_rep1_count, c_rep2_count, c_counts, c_middle_telo_count = count_telo_rep(seq, n)
            # print(f"walk: {i}, c_rep1_count: {c_rep1_count}, c_rep2_count: {c_rep2_count}")
            rep1_counts.append(c_rep1_count); rep2_counts.append(c_rep2_count); counts.extend(c_counts); middle_telo_counts.append(c_middle_telo_count)

    print(f"rep1 sum: {sum(rep1_counts)}, rep2 sum: {sum(rep2_counts)}")

    # Calculate percentiles
    x = np.arange(len(rep1_counts))
    rep1_counts = np.array(rep1_counts); rep2_counts = np.array(rep2_counts)
    percentiles_y1 = [np.percentile(rep1_counts, p) for p in [50, 75, 90]]
    percentiles_y2 = [np.percentile(rep2_counts, p) for p in [50, 75, 90]]

    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, rep1_counts, label=f'{rep1} counts', marker='o')
    plt.plot(x, rep2_counts, label=f'{rep2} counts', marker='o')

    # Add percentile lines for Array 1
    for p, percentile in zip([50, 75, 90], percentiles_y1):
        plt.axhline(y=percentile, color='r', linestyle='--', label=f'rep1 {p}th Percentile')

    # Add percentile lines for Array 2
    for p, percentile in zip([50, 75, 90], percentiles_y2):
        plt.axhline(y=percentile, color='b', linestyle='--', label=f'rep2 {p}th Percentile')

    plt.title(f"genome: {name}, min rep count threshold: {min_count_threshold}")
    plt.xlabel('Walk')
    plt.ylabel('n rep regions')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'graphs/telomere/{name}_rep_regions_in_walk.png')
    plt.clf()


    # second plot on number of repetitive regions
    counts.sort()
    x = np.arange(len(counts))
    counts = np.array(counts)
    percentiles_y1 = [np.percentile(counts, p) for p in [95, 99]]
    print("rep region count percentiles:", percentiles_y1)
    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, counts, label=f'repetitive region counts', marker='o')

    # Add percentile lines for Array 1
    for p, percentile in zip([95, 99], percentiles_y1):
        plt.axhline(y=percentile, color='r', linestyle='--', label=f'rep1 {p}th Percentile')

    plt.ylabel('n rep region is repeated')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'graphs/telomere/{name}_rep_region_count.png')
    plt.clf()


    # third scatter plot for middle walks
    x = np.arange(len(middle_telo_counts))
    plt.scatter(x, middle_telo_counts)
    plt.title(f"genome: {name}, min rep count threshold: {min_count_threshold}")
    plt.xlabel('Walk ID')
    plt.ylabel('n telo regions in middle of walk')
    plt.savefig(f'graphs/telomere/{name}_middle_telo_count.png')
    plt.clf()

        
n_repeats = {
    'arab' : 100,
    'chicken' : 1000,
    'mouse' : 500,
    'chm13' : 200
}
for name in ['arab', 'chicken', 'mouse', 'chm13']:
    analyse_telomere_2(name, 10, n_repeats[name])

# For telomere walk generation
REP1, REP2 = 'TTAGGG', 'CCCTAA' # Repetitive regions used for identifying telomeric sequences
REP_THRESHOLD = 0.05 # % of seq that must contain the above rep regions for it to be classified as telomeric
SEARCH_REGION_P, MAX_SEARCH_REGION_LEN = 0.1, 30000 # Size of start and end region of walk to search

def get_telo_ref(name):
    """
    Generates telomere info for old walks.

    telo_ref = {
        walk_id_1 : {
            'start' : '+', '-', or None,
            'end' : '+', '-', or None
        },
        walk_id_2 : { ... },
        ...
    }
    """

    graph_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl"
    g = dgl.load_graphs(graph_path)[0][0]
    edges = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This is from Martin's script)
    for idx, (src, dst) in enumerate(zip(g.edges()[0], g.edges()[1])):
        src, dst = src.item(), dst.item()
        edges[(src, dst)] = idx

    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl", "rb") as f:
        old_walks = pickle.load(f)
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl", "rb") as f:
        n2s = pickle.load(f)

    contigs = []
    for i, walk in enumerate(old_walks):
        prefixes = [(src, g.edata['prefix_length'][edges[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]
        res = []
        for (src, prefix) in prefixes:
            seq = str(n2s[src])
            res.append(seq[:prefix])
        contig = Seq.Seq(''.join(res) + str(n2s[walk[-1]]))  # TODO: why is this map here? Maybe I can remove it if I work with strings
        contig = SeqIO.SeqRecord(contig)
        contig.id = f'contig_{i+1}'
        contig.description = f'length={len(contig)}'
        contigs.append(contig)

    print("Generating telomere info...")
    def count_rep_region(seq, targ):
        count, start = 0, 0
        while True:
            start = seq.find(targ, start)  # Find next occurrence
            if start == -1:  # If no more occurrences are found
                break
            count += 1  # Increment the count
            start += len(targ)  # Move past this occurrence
        return count

    telo_ref, percentages = {}, []
    for i, contig in enumerate(contigs):
        curr_res = {'start':None, 'end':None}
        seq = str(contig.seq)
        len_seq = int(contig.description.split("=")[1])
        
        for j in ['start', 'end']:
            cutoff = int(min(MAX_SEARCH_REGION_LEN, len_seq//(1/SEARCH_REGION_P)))
            c_seq = seq[:cutoff] if j == 'start' else seq[-cutoff:]
            rep1_count, rep2_count = count_rep_region(c_seq, REP1), count_rep_region(c_seq, REP2)
            if rep1_count > rep2_count:
                c_percentage = (rep1_count*6)/cutoff
                percentages.append(c_percentage)
                if c_percentage >= REP_THRESHOLD: curr_res[j] = '+'
            else:
                c_percentage = (rep2_count*6)/cutoff
                percentages.append(c_percentage)
                if c_percentage >= REP_THRESHOLD: curr_res[j] = '-'

        telo_ref[i] = curr_res

    start_count, end_count, both_count = 0, 0, 0
    for v in telo_ref.values():
        if v['start'] and v['end']:
            if v['start'] == v['end']: print("WARNING: Found same telomeric region in start and end of old walk!")
            both_count += 1
        if v['start']:
            start_count += 1
        if v['end']:
            end_count += 1
    print(f"Start telo walk count: {start_count}, End telo walk count: {end_count}, Both telo walk count: {both_count}")

    percentages.sort()
    p50 = np.percentile(percentages, 50)
    p75 = np.percentile(percentages, 75)
    p90 = np.percentile(percentages, 90)

    # Plotting the line graph
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, label=f'{name} rep region %s', marker='o')
    plt.axhline(y=p50, color='green', linestyle='--', label='50th Percentile (Median)')
    plt.axhline(y=p75, color='orange', linestyle='--', label='75th Percentile')
    plt.axhline(y=p90, color='red', linestyle='--', label='90th Percentile')

    # Adding labels and title
    plt.title(f'{name} rep region %s')
    plt.ylabel('%')
    plt.legend()
    plt.grid()
    plt.savefig(f"graphs/telomere/{name}_walks.png")

    return telo_ref

# get_telo_ref("chm13")

def check_walks_for_dup(name, walks_path):
    print(f"\n=== CHECKING DUPS FOR {name} ===\n")
    with open(walks_path, "rb") as f:
        walks = pickle.load(f)

    print("Number of walks:", len(walks))
    for walk in walks:
        if len(walk) != len(set(walk)): print("dup found!")

# for name in ['chm13', 'mouse', 'chicken', 'arab']:
#     walks_path = "/mnt/sod2-project/csb4/wgs/martin/assemblies/arabidopsis/walks.pkl"
#     # walks_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl"
#     check_walks_for_dup(name, walks_path)
# for chr in [1,5,10,18,19]:
#     for v in ['m', 'p']:
#         walks_path = f"/mnt/sod2-project/csb4/wgs/martin/assemblies/chr_{chr}_synth_{v}/walks.pkl"
#         check_walks_for_dup(chr, walks_path)

def compare_arab_walks():
    with open("pkls/arab_walks_default.pkl", "rb") as d, open("pkls/arab_walks_telomere.pkl", "rb") as t, open("pkls/arab_telo_ref.pkl", "rb") as tr:
        default_walks, telomere_walks, telo_ref = pickle.load(d), pickle.load(t), pickle.load(tr)

    print(telo_ref, "\n")

    default_walks.sort(); telomere_walks.sort()
    print(default_walks, "\n")
    print(telomere_walks)

# compare_arab_walks()