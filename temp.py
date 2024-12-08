import dgl, gc, gzip, mmap, os, pickle, random, subprocess, sqlite3
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import Seq, SeqIO, bgzf
from tqdm import tqdm

from decoding_paf import AdjList, Edge

HAPLOID_TEST_REF = {
    'chm13' : '/mnt/sod2-project/csb4/wgs/martin/genome_references/chm13_v11/chm13_full_v1_1.fasta',
    'arab' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/arabidopsis/latest/GWHBDNP00000000.1.genome.fasta',
    'mouse' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/mus_musculus/mmusculus_GRCm39.fna',
    'chicken' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/bGalGal1/maternal/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna',
    'maize-50p' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/zmays_Mo17/zmays_Mo17.fasta',
    'maize' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/zmays_Mo17/zmays_Mo17.fasta'
}

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

        
# n_repeats = {
#     'arab' : 100,
#     'chicken' : 1000,
#     'mouse' : 500,
#     'chm13' : 200
# }
# for name in ['arab', 'chicken', 'mouse', 'chm13']:
#     analyse_telomere_2(name, 10, n_repeats[name])

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

def analyse_telomeres_99():
    refs = {
        'arab' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/arabidopsis/latest/GWHBDNP00000000.1.genome.fasta',
        'chicken' : '/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/references/bGalGal1/maternal/GCF_016699485.2_bGalGal1.mat.broiler.GRCg7b_genomic.fna',
        'mouse' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/mus_musculus/SRR11606870.fastq',
        'chm13' : '/mnt/sod2-project/csb4/wgs/martin/genome_references/chm13_v11/chm13_full_v1_1.fasta'
    }

    for name in ['chm13', 'arab', 'chicken', 'mouse']:
        print(f"\nANALYSING TELOMERE TYPE FOR {name} ===")
        path = refs[name]
        if path.endswith('gz'):
            if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
                filetype = 'fasta'
            elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
                filetype = 'fastq'
        else:
            if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
                filetype = 'fasta'
            elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
                filetype = 'fastq'

        with open(path, 'rt') as h:
            rows = SeqIO.parse(h, filetype)
            start_count_1s, start_count_2s, end_count_1s, end_count_2s, = [], [], [], []
            for i, row in enumerate(rows):
                seq = str(row.seq)
                cutoff = int(len(seq)*0.01)
                start, end = seq[:cutoff], seq[-cutoff:]
                start_count_1, start_count_2 = start.count('TTAGGG'), start.count('CCCTAA')
                end_count_1, end_count_2 = end.count('TTAGGG'), end.count('CCCTAA')
                # print(f"i: {i}, start_count_1: {start_count_1}, start_count_2: {start_count_2}, end_count_1: {end_count_1}, end_count_2: {end_count_2}")
                start_count_1s.append(start_count_1); start_count_2s.append(start_count_2); end_count_1s.append(end_count_1); end_count_2s.append(end_count_2)

        # Calculate correlation coefficients
        corr_AB = np.corrcoef(start_count_1s, end_count_1s)[0, 1]
        corr_AC = np.corrcoef(start_count_1s, end_count_2s)[0, 1]
        corr_DE = np.corrcoef(start_count_2s, end_count_2s)[0, 1]
        corr_DF = np.corrcoef(start_count_2s, end_count_1s)[0, 1]

        # Create scatter plots
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
        plt.scatter(start_count_1s, end_count_1s, color='blue', label=f'+ve vs +ve, Corr: {corr_AB:.2f}')
        plt.scatter(start_count_1s, end_count_2s, color='red', label=f'+ve vs -ve, Corr: {corr_AC:.2f}')
        plt.xlabel('start count 1s')
        plt.ylabel('Values')
        plt.legend(loc='best')
        plt.grid(True)

        # Subplot 2: Lists D, E, and F
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
        plt.scatter(start_count_2s, end_count_2s, color='blue', label=f'-ve vs -ve, Corr: {corr_DE:.2f}')
        plt.scatter(start_count_2s, end_count_1s, color='red', label=f'-ve vs +ve, Corr: {corr_DF:.2f}')
        plt.xlabel('start count 2s')
        plt.ylabel('Values')
        plt.legend(loc='best')
        plt.grid(True)

        # Save the figure
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(f'graphs/telomere/{name}_telo_types.png')  # Save as PNG file
        plt.show()  # Show the plots

# analyse_telomeres_99()

def telomere_extraction(name, fasta_path, seqtk_path='../GitHub/seqtk/seqtk'):
    print(f"\n=== RUNNING TELOMERE EXTRACTION FOR {name} ===\n")
    read_to_seq = {}
    with open(fasta_path, 'rt') as handle:
        rows = SeqIO.parse(handle, 'fasta')
        for row in rows:
            id, seq = row.description.split()[0], str(row.seq)
            read_to_seq[id] = seq

    if name in ["maize", "maize-50p", "arab"]:
        rep1, rep2 = 'TTTAGGG', 'CCCTAAA'
    else:
        rep1, rep2 = 'TTAGGG', 'CCCTAA'
    seqtk_cmd_rep1 = f"{seqtk_path} telo -m {rep1} {fasta_path}"
    seqtk_cmd_rep2 = f"{seqtk_path} telo -m {rep2} {fasta_path}"
    seqtk_res_rep1 = subprocess.run(seqtk_cmd_rep1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seqtk_res_rep2 = subprocess.run(seqtk_cmd_rep2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    seqtk_res_rep1 = seqtk_res_rep1.stdout.split("\n"); seqtk_res_rep1.pop()
    seqtk_res_rep2 = seqtk_res_rep2.stdout.split("\n"); seqtk_res_rep2.pop()
    
    n_rep1, n_rep2 = 0, 0
    for row in seqtk_res_rep1:
        row_split = row.split("\t")
        walk_id, start, end = row_split[0], int(row_split[1]), int(row_split[2])-1
        c_seq = read_to_seq[walk_id][start:end]
        rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
        if rep1_count > rep2_count: 
            n_rep1 += 1
        else:
            n_rep2 += 1
    for row in seqtk_res_rep2:
        row_split = row.split("\t")
        walk_id, start, end = row_split[0], int(row_split[1]), int(row_split[2])-1
        c_seq = read_to_seq[walk_id][start:end]
        rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
        if rep1_count > rep2_count: 
            n_rep1 += 1
        else:
            n_rep2 += 1

    print(f"telomere extraction done. n_rep1: {n_rep1}, n_rep2: {n_rep2}")

def test_are_walks_order_the_same(name):
    print(f"=== CHECKING FOR {name} ===")
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl", 'rb') as f:
        n2s = pickle.load(f)
    graph = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl")[0][0]
    
    edges_full = {}
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_full[(src, dst)] = idx

    fasta_seqs = []
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/0_assembly.fasta", 'rt') as f:
        rows = SeqIO.parse(f, 'fasta')
        for i, record in enumerate(tqdm(rows, ncols=120)):
            seq = str(record.seq)
            fasta_seqs.append(seq)

    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl", 'rb') as f:
        walks = pickle.load(f)
        for i, walk in enumerate(walks):
            prefixes = [(src, graph.edata['prefix_length'][edges_full[src,dst]]) for src, dst in zip(walk[:-1], walk[1:])]

            res = []
            for (src, prefix) in prefixes:
                seq = str(n2s[src])
                res.append(seq[:prefix])

            contig = ''.join(res) + str(n2s[walk[-1]]) # TODO: why is this map here? Maybe I can remove it if I work with strings
            if contig != fasta_seqs[i]: print("DIFFERENT! i:", i)

# for n in ['mouse']:
#     test_are_walks_order_the_same(n)

from decoding_paf import chop_walks_seqtk
def run_chop_walks_seqtk(name):
    print(f"=== RUNNING FOR {name} ===")
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{name}/walks.pkl", 'rb') as f:
        walks = pickle.load(f)
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_n2s.pkl", 'rb') as f:
        n2s = pickle.load(f)
    old_graph = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl")[0][0]
    seqtk_path = "../GitHub/seqtk/seqtk"
    telo_motif_ref = {
        'arab' : ("TTTAGGG", "CCCTAAA"),
        'chicken' : ("TTAGGG", "CCCTAA"),
        'mouse' : ("TTAGGG", "CCCTAA"),
        'chm13' : ("TTAGGG", "CCCTAA"),
        'maize-50p' : ("TTTAGGG", "CCCTAAA"),
        'maize' : ("TTTAGGG", "CCCTAAA")
    }
    chop_walks_seqtk(walks, n2s, old_graph, telo_motif_ref[name][0], telo_motif_ref[name][1], seqtk_path)

# for n in ['arab', 'chicken', 'mouse']:
#     run_chop_walks_seqtk(n)

def run_quast(name, type):
    print(f"=== RUNNING QUAST FOR {name}, {type} ===")
    asm_dir_name = 'default' if type == "GNNome" else type 
    save_path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/analysis/quast/{type}/{name}/'
    cmd = f"quast /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/{asm_dir_name}/{name}/0_assembly.fasta -r {HAPLOID_TEST_REF[name]} -o {save_path} -t 16"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# for n in ['mouse', 'maize']:
#     run_quast(n, 'postprocessed')

def run_compleasm(name, type):
    print(f"=== RUNNING COMPLEASM FOR {name}, {type} ===")
    asm_dir_name = 'default' if type == "GNNome" else type 
    save_path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/analysis/compleasm/{type}/{name}/'
    cmd = f"python ../GitHub/compleasm_kit/compleasm.py run -a /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/{asm_dir_name}/{name}/0_assembly.fasta -o {save_path} --autolineage -t 16"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # print(cmd)
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# for n in ['mouse', 'maize']:
#     run_compleasm(n, 'postprocessed')

def read_seed_minigraphs(names):
    for n in names:
        for seed in range(6):
            print(f"\n=== {n}, seed {seed} ===")
            file_name = "0_minigraph_latest.txt" if n == "arabidopsis_p022_l0" else "0_minigraph.txt"
            with open(f'/mnt/sod2-project/csb4/wgs/lovro/gnnome_assembly/BEST_ARCH_RESULTS/23-08-21_60xMASK-symloss_h64_drop020_seed{seed}/{n}/reports/{file_name}') as f:
                report = f.read()
                print(report)

def compare_gfas(name):
    print(f"=== RUNNING FOR {name} ===")
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/default_{name}_r2n.pkl", 'rb') as f:
        r2n = pickle.load(f)
    graph = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/default/{name}.dgl")[0][0]
    
    prefix_len_mapping = {}
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        # print(src, dst)
        prefix_len_mapping[(src.item(), dst.item())] = int(graph.edata['prefix_length'][idx].item())

    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{name}.bp.p_ctg.gfa") as f:
        rows = f.readlines()
        contigs = defaultdict(list)
        for i, row in enumerate(rows):
            row = row.strip().split()
            if row[0] != "A": continue
            contigs[row[1]].append(row)

    missing_nodes, missing_edges = set(), set()
    base_mismatch_count, base_mismatch_abs = 0, 0
    for contig, reads in contigs.items():
        reads = sorted(reads, key=lambda x:int(x[2]))
        # print(reads)
        for i in range(len(reads)-1):
            curr, next = reads[i], reads[i+1]
        
            if next[4] not in r2n or curr[4] not in r2n:
                if curr[4] not in r2n:
                    missing_nodes.add(curr[4])
                if next[4] not in r2n:
                    missing_nodes.add(next[4])
                missing_edges.add((curr[4], next[4]))
                continue

            src_node = r2n[curr[4]][0] if curr[3] == "+" else r2n[curr[4]][1]
            dst_node = r2n[next[4]][0] if next[3] == "+" else r2n[next[4]][1]
        if (src_node, dst_node) not in prefix_len_mapping:
            missing_edges.add((curr[4], next[4]))
        else:
            curr_prefix = int(next[2])-int(curr[2])
            if curr_prefix != prefix_len_mapping[(src_node, dst_node)]:
                base_mismatch_count += 1
                base_mismatch_abs += abs(prefix_len_mapping[(src_node, dst_node)]-curr_prefix)
                # print(f"Prefix length mismatch found! Old: {prefix_len_mapping[(src_node, dst_node)]}, New: {curr_prefix}")

    print(f"Finish parsing new GFA. N Contigs: {len(contigs)}, Missing nodes: {len(missing_nodes)}, Missing edges: {len(missing_edges)}, Base Mismatch: {base_mismatch_abs/base_mismatch_count}")

    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{name}.ovlp.paf") as f:
        rows = f.readlines()
        for i, row in tqdm(enumerate(rows), ncols=120):
            row = row.strip().split()
            missing_edges.discard((row[0], row[5]))
            if not missing_edges: break

    if missing_edges:
        print("Number of missing edges after PAF:", len(missing_edges))
        count = 0
        for e in missing_edges:
            if e[0] in missing_nodes or e[1] in missing_nodes:
                count += 1
        print("Number with a missing node:", count)
    else:
        print("All missing edges found!")

from difflib import SequenceMatcher

def assemble_hifi_gfa(name):
    def find_most_similar(target, string_set):
        most_similar = None
        highest_similarity = 0

        for s in string_set:
            similarity = SequenceMatcher(None, target, s).ratio()
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar = s

        return most_similar, highest_similarity


    print(f"=== RUNNING FOR {name} ===")
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{name}_fasta_data.pkl", 'rb') as f:
        fasta_data = pickle.load(f)

    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{name}.bp.p_ctg.gfa") as f:
        rows = f.readlines()
        contigs = defaultdict(list)
        for i, row in enumerate(rows):
            row = row.strip().split()
            if row[0] != "A": continue
            contigs[row[1]].append(row)

    hifi_gfa_seqs = set()
    with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{name}.p_ctg.fa", 'rt') as f:
        rows = SeqIO.parse(f, 'fasta')
        for i, record in enumerate(tqdm(rows, ncols=120)):
            seq = str(record.seq)
            hifi_gfa_seqs.add(seq)

    assembled_seqs = set()
    for contig, reads in contigs.items():
        reads = sorted(reads, key=lambda x:int(x[2]))
        # print(reads)
        cs = ""
        for i in range(len(reads)-1):
            curr, next = reads[i], reads[i+1]

            curr_prefix = int(next[2])-int(curr[2])
            curr_seq = fasta_data[curr[4]][0] if curr[3] == '+' else fasta_data[curr[4]][1]
            curr_seq = curr_seq[int(curr[5]):int(curr[6])]
            cs += curr_seq[:curr_prefix+1]
        
        curr = reads[-1]
        curr_seq = fasta_data[curr[4]][0] if curr[3] == '+' else fasta_data[curr[4]][1]
        curr_seq = curr_seq[int(curr[5]):int(curr[6])]
        cs += curr_seq
        assembled_seqs.add(cs)

    for i in assembled_seqs:
        most_similar, highest_similarity = find_most_similar(i, hifi_gfa_seqs)
        print("length assembled seq:", len(i), "length hifi seq:", len(most_similar), "similarity:", highest_similarity)

    print(len(hifi_gfa_seqs), len(assembled_seqs))
    unique_in_set1 = hifi_gfa_seqs.difference(assembled_seqs)
    print("Len Unique in Set 1:", len(unique_in_set1))

def find_dupes_in_gfa():
    with open("/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.bp.p_ctg.gfa") as f:
        rows = f.readlines()

    contigs = defaultdict(list)
    for row in rows:
        row = row.strip().split()
        if row[0] != "A": continue
        contigs[row[1]].append(row)

    edges = set()
    for contig, reads in contigs.items():
        reads = sorted(reads, key=lambda x:int(x[2])) # sort by order in contig
        for i in range(len(reads)-1):
            curr_row, next_row = reads[i], reads[i+1]
            curr_edge = (curr_row[4], next_row[4])
            if curr_edge in edges:
                print("duplicate found!")
            else:
                edges.add(curr_edge)

R2T = {}

def parse_record_p(record):
    if not R2T: print("Global r2t not set!")
    return R2T[record.id] != 'm', record

def parse_record_m(record):
    if not R2T: print("Global r2t not set!")
    return R2T[record.id] != 'p', record

def split_fasta(yak_path, reads_path, save_path, name, which="b"):
    global R2T; R2T = {}
    n_ps, n_ms, n_bs = 0, 0, 0
    with open(yak_path) as f:
        rows = f.readlines()
        for row in rows:
            row = row.strip().split()
            if row[1] == 'm':
                R2T[row[0]] = 'm'
                n_ms += 1
            elif row[1] == 'p':
                R2T[row[0]] = 'p'
                n_ps += 1
            elif row[1] == '0':
                R2T[row[0]] = 'b'
                n_bs += 1
            elif row[1] == 'a':
                R2T[row[0]] = 'b'
                n_bs += 1
            else:
                print("Unrecognised type!")
    print(f"Finished reading yak triobinning. n ps: {n_ps}, n ms: {n_ms}, n_bs: {n_bs}")

    if which != "m":
        print("Starting to filter for paternal...")
        p_contigs = []
        with gzip.open(reads_path, 'rt') as f:
            rows = SeqIO.parse(f, 'fastq')
            with Pool(15) as pool:
                results = pool.imap_unordered(parse_record_p, rows, chunksize=50)
                for to_include, record in tqdm(results, ncols=120):
                    if to_include: p_contigs.append(record)
        print("Finished parsing for paternal. Number of reads:", len(p_contigs))
        SeqIO.write(p_contigs, f"{save_path}{name}/paternal/{name}_yak_P.fasta", 'fasta')

        del p_contigs
        gc.collect()

    if which != "p":
        print("Starting to filter for maternal...")
        m_contigs = []
        with gzip.open(reads_path, 'rt') as f:
            rows = SeqIO.parse(f, 'fastq')
            with Pool(15) as pool:
                results = pool.imap_unordered(parse_record_m, rows, chunksize=50)
                for to_include, record in tqdm(results, ncols=120):
                    if to_include: m_contigs.append(record)
        print("Finished parsing for maternal. Number of reads:", len(m_contigs))
        SeqIO.write(m_contigs, f"{save_path}{name}/maternal/{name}_yak_M.fasta", 'fasta')

def rename_files(folder, old_name, new_name):
    for filename in os.listdir(folder):
        if filename.startswith(old_name):
            # Build the new filename
            new_filename = new_name + filename[len(old_name):]
            # Get full paths for renaming
            old_file = os.path.join(folder, filename)
            new_file = os.path.join(folder, new_filename)
            # Rename the file
            os.rename(old_file, new_file)

from pyfaidx import Fasta
from concurrent.futures import ThreadPoolExecutor, as_completed

def dummy(row):
    r2s = Fasta(path)
    row = row.strip().split()
    read1, read2 = row[0], row[5]
    return str(r2s[read1][:]), str(-r2s[read2][:])

def test_pyfaidx():
    with open('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ovlp.paf') as f:
        rows = f.readlines()
    print('hi')

    # global r2s 
    # r2s = Fasta('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ec.fa')
    # with ThreadPoolExecutor(max_workers=40) as executor:
    #     futures = [executor.submit(dummy, row) for row in rows]
    #     for future in tqdm(as_completed(futures), total=len(futures), ncols=120):
    #         a, b = future.result()
    
    global path
    path = '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ec.fa'
    with Pool(40) as pool:
        results = pool.imap_unordered(dummy, iter(rows), chunksize=160)
        for code, data in tqdm(results, total=len(rows), ncols=120):
            a = 1

def test_pyfaidx2():
    r2s = Fasta('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/bonobo_ont/temp.fasta')
    read = '1621606c-3050-4346-99e2-0500b61e1c37'
    print(r2s[read][:])

def convert_fastq_to_fasta(fastq_path, fasta_path):
    SeqIO.convert(fastq_path, 'fastq', fasta_path, 'fasta')

def yak_metrics(save_path, yak1, yak2, yak_path):
    """
    IMPT: asm_metrics have to be run before this to generate the assembly!
    
    Yak triobinning result files have following info:
    C       F  seqName     type      startPos  endPos    count
    C       W  #switchErr  denominator  switchErrRate
    C       H  #hammingErr denominator  hammingErrRate
    C       N  #totPatKmer #totMatKmer  errRate
    """
    print("Running yak trioeval...")
    save_file = save_path+"phs.txt"
    cmd = f'{yak_path} trioeval -t16 {yak1} {yak2} {save_path}0_assembly.fasta > {save_file}'.split()
    with open(save_file, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f)
    p.wait()

    switch_err, hamming_err = None, None
    with open(save_file, 'r') as file:
        # Read all the lines and reverse them
        lines = file.readlines()
        reversed_lines = reversed(lines)
        for line in reversed_lines:
            if line.startswith('W'):
                switch_err = float(line.split()[3])
            elif line.startswith('H'):
                hamming_err = float(line.split()[3])
            if switch_err is not None and hamming_err is not None:
                break

    print(f"YAK Switch Err: {switch_err*100:.4f}%, YAK Hamming Err: {hamming_err*100:.4f}%")

def evaluate_baseline_yak():
    for genome in ["hg002_d_20x_p", "hg002_d_20x_m", "hg002_d_20x_scaf_p", "hg002_d_20x_scaf_m", "bonobo_d_20x_p", "bonobo_d_20x_m", "bonobo_d_20x_scaf_p", "bonobo_d_20x_scaf_m"]:
        print(f"=== RUNNING FOR {genome} ===\n")
        if genome.startswith('hg002'):
            yak1, yak2 = '/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/pat.yak', '/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/mat.yak'
        else:
            yak1, yak2 = '/mnt/sod2-project/csb4/wgs/martin/genome_references/mPanPan1_v2/panpan2.yak', '/mnt/sod2-project/csb4/wgs/martin/genome_references/mPanPan1_v2/panpan3.yak'

        save_path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/baseline/hifiasm/{genome}/'
        yak_metrics(save_path, yak1, yak2, '/home/stumanuel/GitHub/yak/yak')

def parse_read(read):
    description = read.description.split()
    id = description[0]
    seqs = (str(read.seq), str(Seq.Seq(read.seq).reverse_complement()))
    train_desc = read.description

    return id, seqs, train_desc

def parse_fasta(path):
    print("Parsing FASTA...")
    if path.endswith('gz'):
        if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
            filetype = 'fasta'
        elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
            filetype = 'fastq'
    else:
        if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
            filetype = 'fasta'
        elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
            filetype = 'fastq'

    data = {}
    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as handle:
        rows = SeqIO.parse(handle, filetype)

        with Pool(15) as pool:
            results = pool.imap_unordered(parse_read, rows, chunksize=50)
            for id, seqs, train_desc in tqdm(results, ncols=120):
                data[id] = seqs

    return data


def test_mmap_read(row):
    row = row.strip().split()
    return row[0], row[5]

# def test_mmap():
#     r2s = parse_fasta('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ec.fa')
#     file_name = "temp.pkl"
#     with open(file_name, "wb") as f:
#         pickle.dump(r2s, f)
#     with open('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ovlp.paf') as f:
#         rows = f.readlines()

#     global loaded
#     with open(file_name, "rb") as f:
#         mmaped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
#         loaded = pickle.loads(mmaped_file)

#         with Pool(20) as pool:
#             results = pool.imap_unordered(test_mmap_read, iter(rows), chunksize=160)
#             for read1, read2, seq1, seq2 in tqdm(results, total=len(rows), ncols=120):
#                 assert r2s[read1] == seq1, "Read1 mismatch!"
#                 assert r2s[read2] == seq2, "Read2 mismatch!"

#     mmaped_file.close()
#     os.remove(file_name)

class TempDB():
    def __init__(self):
        db_name = f'{random.randint(1,9999999)}.db'
        self.db = db_name
        conn = sqlite3.connect(db_name)
        conn.execute('PRAGMA synchronous=OFF;')  # Disable synchronous mode
        conn.execute('PRAGMA journal_mode=MEMORY;')  # Use in-memory journaling
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS r2s (
            read TEXT PRIMARY KEY,
            seq TEXT,
            seq_rev TEXT
        )
        ''')
        conn.commit()
        conn.close()

    def add(self, read, seqs):
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()

        # Insert or replace the key-value pair
        cur.execute('''
        INSERT OR REPLACE INTO r2s (read, seq, seq_rev)
        VALUES (?, ?, ?)
        ''', (read, seqs[0], seqs[1]))

        conn.commit()
        conn.close()

    def add_batch(self, data):
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        
        conn.execute('BEGIN TRANSACTION')

        statement = '''
        INSERT OR REPLACE INTO r2s (read, seq, seq_rev)
        VALUES (?, ?, ?)
        '''
        cur.executemany(statement, data)

        # for read, seqs in data:
        #     cur.execute('''
        #     INSERT OR REPLACE INTO r2s (read, seq, seq_rev)
        #     VALUES (?, ?, ?)
        #     ''', (read, seqs[0], seqs[1]))

        conn.commit()
        conn.close()

    def get(self, read):
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()

        # Fetch the value associated with the key
        cur.execute('SELECT seq, seq_rev FROM r2s WHERE read = ?', (read,))
        result = cur.fetchone()  # Returns a tuple like ('value1',) or None if not found

        conn.close()
        if result:
            return result
        else:
            raise ValueError("Missing read!")
        
    def close(self):
        if os.path.exists(self.db):
            os.remove(self.db)

def parse_fasta_2(path):
    print("Parsing FASTA...")
    if path.endswith('gz'):
        if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
            filetype = 'fasta'
        elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
            filetype = 'fastq'
    else:
        if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
            filetype = 'fasta'
        elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
            filetype = 'fastq'

    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as handle:
        rows = SeqIO.parse(handle, filetype)

        curr_batch = []
        with Pool(15) as pool:
            results = pool.imap_unordered(parse_read, rows, chunksize=50)
            for idx, (read, seqs, train_desc) in tqdm(enumerate(results), ncols=120):
                curr_batch.append((read, seqs[0], seqs[1]))
                if idx % 100000 == 0:
                    R2S.add_batch(curr_batch)
                    curr_batch = []

            R2S.add_batch(curr_batch)

    return

def test_custom_db():
    global R2S
    R2S = TempDB()

    parse_fasta_2('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ec.fa')
    r2s_local = parse_fasta('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ec.fa')

    with open('/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/arab/arab.ovlp.paf') as f:
        rows = f.readlines()
    with Pool(20) as pool:
        results = pool.imap_unordered(test_mmap_read, iter(rows), chunksize=160)
        for read1, read2 in tqdm(results, total=len(rows), ncols=120):
            pass 
            # assert R2S.get(read1)[0] == r2s_local[read1][0], "Read1 mismatch!"
            # assert R2S.get(read2)[0] == r2s_local[read2][0], "Read2 mismatch!"

    R2S.close()

if __name__ == "__main__":
    # for n in ['mouse', 'arab', 'chicken', 'chm13', 'maize-50p']:
    #     # walks_fasta_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/res/default/{n}/0_assembly.fasta"
    #     telomere_extraction(n, HAPLOID_TEST_REF[n])

    # read_seed_minigraphs(["arabidopsis_p022_l0", "bGalGal1_maternal_0.5_30x", "mus_musculus", "full_chm13"])

    # for n in ['arab']:
    #    assemble_hifi_gfa(n)

    # find_dupes_in_gfa()

    # split_fasta(
    #     yak_path="/mnt/sod2-project/csb4/wgs/martin/real_diploid_data/hifi_data/bonobo_20.triobin",
    #     reads_path="/mnt/sod2-project/csb4/wgs/martin/real_diploid_data/hifi_data/bonobo_c20/full_reads/bonobo_full_0.fastq.gz",
    #     save_path="/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/",
    #     name="bonobo_20x",
    #     which="b"
    # )

    # rename_files(
        # folder="/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/bonobo_30x_m",
        # old_name="bonobo_m",
        # new_name="bonobo_30x_m"
    # )

    convert_fastq_to_fasta(
        fastq_path='/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/tomato_ont/tomato_ont.ec.fq',
        fasta_path='/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/tomato_ont/tomato_ont.ec.fa'
    )
