import dgl, gc, glob, gzip, mmap, os, pickle, random, re, shutil, subprocess, sqlite3, yaml
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import seaborn as sns
from Bio import SeqIO, bgzf
from Bio.Seq import Seq
from tqdm import tqdm
import networkx as nx
from datetime import datetime

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

def convert_fastq_to_fasta_ec(name):
    fastq_path=f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{name}/{name}.ec.fq'
    fasta_path=f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{name}/{name}.ec.fa'
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


def test_mmap_read(row):
    row = row.strip().split()
    return row[0], row[5]

def run_quast(name, type='res'):
    print(f"\n=== RUNNING QUAST FOR {name}, type: {type} ===")
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    save_path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}'
    cmd = f"quast {save_path}/0_assembly.fasta -r {config['genome_info'][name]['paths']['ref']} -o {save_path}/quast -t 16"
    if os.path.exists(save_path+"/quast"): shutil.rmtree(save_path+"/quast")
    os.makedirs(save_path+"/quast")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(res.stderr)
    with open(save_path+"/quast/report.txt") as f:
        report = f.read()
        print(report)

reads = {
    'arab' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/arabidopsis_new/PacBio_HiFi/CRR302668_p0.22.fastq.gz',
    'chicken' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/gallus_gallus/HiFi/mat_0.5_30x.fastq.gz',
    'chm13' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/PacBio_HiFi/SRR11292120_3_subreads.fastq.gz',
    'maize' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/zmays_Mo17/HiFi/zmays_Mo17-HiFi.fastq.gz',
    'mouse' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/mus_musculus/SRR11606870.fastq',
    'hg002_d_20x_scaf' : '/mnt/sod2-project/csb4/wgs/martin/real_diploid_data/hifi_data/hg002_c20/full_reads/hg002_v101_full_0.fastq.gz',
    'bonobo_d_20x_scaf' : '/mnt/sod2-project/csb4/wgs/martin/real_diploid_data/hifi_data/bonobo_c20/full_reads/mPanPan1_v2_full_0.fastq.gz',
    'gorilla_d_20x_scaf' : '/mnt/sod2-project/csb4/wgs/martin/real_diploid_data/hifi_data/gorilla_c20/full_reads/mGorGor1_v2_full_0.fastq.gz',
    'arab_ont' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/arab_ont/SRR29061597_1.fastq.gz',
    'fruitfly_ont' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/fruitfly/ont/SRR23215007_1.fastq.gz',
    'tomato_ont' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/tomato/ont/R10.4_40x.noduplex.fastq.gz',
    'hg005_d_ont_scaf' : '/mnt/sod2-project/csb4/wgs/lindehui/EVAL/HG005/11_1_22_R1041_Sheared_HG005_1_Guppy_6.2.11_prom_sup.fastq.gz',
    'hg002_d_ont_scaf' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/hg002/hg002_ont/PAO83395.fastq',
    'gorilla_d_ont_20x_scaf' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/gorilla/gorilla_ont/gorilla_ont.fa',
    'bonobo_d_ont_20x_scaf' : '/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/datasets/bonobo/bonobo_ont/bonobo_ont.fasta'
}

def gen_yak_count(name):
    print("GENERATING YAK COUNT FOR:", name)
    cmd = f'/home/stumanuel/GitHub/yak/yak count -b37 -t16 -o /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{name}/{name}.yak {reads[name]}'
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(res.stderr)
    return

def gen_yak_qv(name, type='res'):
    print("GENERATING YAK QV FOR:", name, type)
    if name.endswith('_p') or name.endswith('_m'):
        yak_name = name[:-2]
    else:
        yak_name = name

    cmd = f'/home/stumanuel/GitHub/yak/yak qv -t16 -p /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{yak_name}/{yak_name}.yak /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/0_assembly.fasta > /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/{name}.qv.txt'
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print(res.stdout)
    return

def read_yak_qv(name, type='res'):
    print(f"READING YAK QV FOR {name}, {type}")
    file = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/{name}.qv.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
        print(lines[-1])

    return

def analyse_t2t(name, type='res'):
    print(f"ANALYSING T2T FOR {name}, {type}")
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    if os.path.exists(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/0_assembly.fasta.seqkit.fai"):
        # This has been run before, delete and re-run
        os.remove(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/0_assembly.fasta.seqkit.fai")
        pattern = os.path.join(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/", "T2T*")
        ftd = glob.glob(pattern)
        for f in ftd:
            os.remove(f)

    motif = config['genome_info'][name]['telo_motifs'][0]
    cmd = f"/home/stumanuel/GitHub/T2T_Sequences/T2T_chromosomes.sh -a /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/0_assembly.fasta -r {config['genome_info'][name]['paths']['ref']} -m {motif} -t 10"
    cwd = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}"
    subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return

def count_t2t(name, type='res'):
    print(f"Number of T2T seqs for {name}, {type}")

    aligned_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/T2T_sequences_alignment_T2T.txt"
    with open(aligned_path, 'r') as f:
        aligned_count = sum(1 for line in f)
    unaligned_path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/{type}/hifiasm/{name}/T2T_sequences_motif_T2T.txt"
    with open(unaligned_path, 'r') as f:
        unaligned_count = sum(1 for line in f)
        
    print(f"Unaligned: {unaligned_count} | Aligned: {aligned_count}")
    
    return

def draw_graph():
    with open('../GAP/misc/temp/temp_graph_nx.pkl', "rb") as f:
        nx_g = pickle.load(f)

    plt.figure(figsize=(30,30))
    nx.draw(nx_g, with_labels=True, node_size=50, font_size=9)
    plt.savefig('../GAP/misc/temp/temp_graph_nx.png')

def get_seqs(id, hifi_r2s, ul_r2s):
    if id in hifi_r2s:
        return str(hifi_r2s[id][:]), str(-hifi_r2s[id][:])
    elif ul_r2s is not None and id in ul_r2s:
        return str(ul_r2s[id][:]), str(-ul_r2s[id][:])
    else:
        raise ValueError("Read not present in seq dataset FASTAs!")

def hifiasm_decoding(paths, motif):
    print(f"Loading files...")
    c2s, links = {}, {}
    with open(paths["gfa"]) as f:
        rows = f.readlines()
        for row in rows:
            row = row.strip().split()
            if row[0] == 'S': c2s[row[1]] = row[2]
            if row[0] == "L": 
                cid1, cid2 = row[1], row[3]
                if cid1 == cid2: continue
                orient1, orient2 = row[2], row[4]
                overlap = int(row[5][:-1])
                if cid1 in links:
                    if overlap > links[cid1][1]:
                        links[cid1] = (cid2, overlap, orient1, orient2)
                else:
                    links[cid1] = (cid2, overlap, orient1, orient2)

    print("Creating graph...")
    g = nx.DiGraph()
    g.add_nodes_from(c2s.keys())
    g.add_weighted_edges_from([(cid1, v[0], v[1]) for cid1, v in links.items()])

    try:
        while True:
            cycle = nx.find_cycle(g, orientation='original')
            # Remove the edge with the minimal weight in the detected cycle
            edge_to_remove = min(cycle, key=lambda x: g[x[0]][x[1]]['weight'])
            g.remove_edge(*edge_to_remove[:2])
            print(f"Removed edge {edge_to_remove} to break the cycle")
    except nx.NetworkXNoCycle:
        print("No cycles left. The graph is acyclic.")

    walks = []
    while g.number_of_nodes() > 0:
        longest_walk = nx.dag_longest_path(g)
        walks.append(longest_walk)
        g.remove_nodes_from(longest_walk)

    print("Creating contigs...")
    contigs = []
    for idx, walk in enumerate(walks):
        c_contig = ""
        for i, cid in enumerate(walk[:-1]):
            edge = links[cid]
            if edge[0] != walk[i+1]: print("wrong edge found.... sigh")
            overlap, orient1 = edge[1], edge[2]
            if orient1 == '-':
                c_contig += c2s[cid][::-1][:-overlap]
            else:
                c_contig += c2s[cid][:-overlap]

        if len(walk) > 1:
            if links[walk[-2]][3] == '+':
                c_contig += c2s[walk[-1]]
            else:
                c_contig += c2s[walk[-1]][::-1]
        else:
            c_contig += c2s[walk[-1]]

        c_contig = Seq.Seq(c_contig)
        c_contig = SeqIO.SeqRecord(c_contig)
        c_contig.id = f'contig_{idx+1}'
        contigs.append(c_contig)

    print(f"Calculating assembly metrics...")
    save_path = "temp_save/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    asm_path = save_path+f"temp_{random.randint(1,9999999)}.fasta"
    SeqIO.write(contigs, asm_path, 'fasta')

    cmd = f'/home/stumanuel/GitHub/minigraph/minigraph -t32 -xasm -g10k -r10k --show-unmap=yes {paths["ref"]} {asm_path}'.split(' ')
    paf = save_path+'asm.paf'
    with open(paf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
    p.wait()

    cmd = f'k8 /home/stumanuel/GitHub/minimap2/misc/paftools.js asmstat {paths["ref"]+".fai"} {paf}'.split()
    report = save_path+"minigraph.txt"
    with open(report, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)
    p.wait()
    with open(report) as f:
        report = f.read()
        print(report)

    shutil.rmtree(save_path)
    return

def test_paf_edges(genome, full_reads_path):
    def chop_walks_seqtk(old_walks, n2s, graph, rep1, rep2, seqtk_path):
        """
        Generates telomere information, then chops the walks. 
        1. I regenerate the contigs from the walk nodes. I'm not sure why but when regenerating it this way it differs slightly from the assembly fasta, so i'm doing it this way just to be safe.
        2. seqtk is used to detect telomeres, then I manually count the motifs in each region to determine if it is a '+' or '-' motif.
        3. The walks are then chopped. When a telomere is found:
            a. If there is no telomere in the current walk, and the walk is already >= twice the length of the found telomere, the telomere is added to the walk and then chopped.
            b. If there is an opposite telomere in the current walk, the telomere is added to the walk and then chopped.
            c. If there is an identical telomere in the current walk, the walk is chopped and a new walk begins with the found telomere.
        """

        # Create a list of all edges
        edges_full = {}  ## I dont know why this is necessary. but when cut transitives some edges are wrong otherwise. (This comment is from Martin's script)
        for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
            src, dst = src.item(), dst.item()
            edges_full[(src, dst)] = idx

        # Regenerate old contigs
        old_contigs, pos_to_node = [], defaultdict(dict)
        for walk_id, walk in enumerate(old_walks):
            seq, curr_pos = "", 0
            for idx, node in enumerate(walk):
                # Preprocess the sequence
                c_seq = str(n2s[node])
                if idx != len(walk)-1:
                    c_prefix = graph.edata['prefix_length'][edges_full[node,walk[idx+1]]]
                    c_seq = c_seq[:c_prefix]

                seq += c_seq
                c_len_seq = len(c_seq)
                for i in range(curr_pos, curr_pos+c_len_seq):
                    pos_to_node[walk_id][i] = node
                curr_pos += c_len_seq
            old_contigs.append(seq)
        
        temp_fasta_name = f'temp_{random.randint(1,9999999)}.fasta'
        with open(temp_fasta_name, 'w') as f:
            for i, contig in enumerate(old_contigs):
                f.write(f'>{i}\n')  # Using index as ID
                f.write(f'{contig}\n')

        # Use seqtk to get telomeric regions
        seqtk_cmd_rep1 = f"{seqtk_path} telo -m {rep1} {temp_fasta_name}"
        seqtk_cmd_rep2 = f"{seqtk_path} telo -m {rep2} {temp_fasta_name}"
        seqtk_res_rep1 = subprocess.run(seqtk_cmd_rep1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        seqtk_res_rep2 = subprocess.run(seqtk_cmd_rep2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if seqtk_res_rep1.returncode != 0: raise RuntimeError(seqtk_res_rep1.stderr.strip())
        if seqtk_res_rep2.returncode != 0: raise RuntimeError(seqtk_res_rep2.stderr.strip())
        seqtk_res_rep1 = seqtk_res_rep1.stdout.split("\n"); seqtk_res_rep1.pop()
        seqtk_res_rep2 = seqtk_res_rep2.stdout.split("\n"); seqtk_res_rep2.pop()

        telo_info = defaultdict(dict)
        for row in seqtk_res_rep1:
            row_split = row.split("\t")
            walk_id, start, end = int(row_split[0]), int(row_split[1]), int(row_split[2])-1
            c_seq = old_contigs[walk_id][start:end]
            rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
            c_rep = rep1 if rep1_count > rep2_count else rep2
            start_node, end_node = pos_to_node[walk_id][start], pos_to_node[walk_id][end]
            if start_node in telo_info[walk_id]:
                print("Duplicate telomere region found 1!")
            else:
                telo_info[walk_id][start_node] = (end_node, c_rep)
        for row in seqtk_res_rep2:
            row_split = row.split("\t")
            walk_id, start, end = int(row_split[0]), int(row_split[1]), int(row_split[2])-1
            c_seq = old_contigs[walk_id][start:end]
            rep1_count, rep2_count = c_seq.count(rep1), c_seq.count(rep2)
            c_rep = rep1 if rep1_count > rep2_count else rep2
            start_node, end_node = pos_to_node[walk_id][start], pos_to_node[walk_id][end]
            if start_node in telo_info[walk_id]:
                print("Duplicate telomere region found 2!")
            else:
                telo_info[walk_id][start_node] = (end_node, c_rep)
        os.remove(temp_fasta_name)

        # Chop walks
        new_walks, telo_ref = [], {}
        for walk_id, walk in enumerate(old_walks):
            curr_ind, curr_walk, curr_telo = 0, [], None
            while curr_ind < len(walk):
                curr_node = walk[curr_ind]
                if curr_node in telo_info[walk_id]:
                    end_node, telo_type = telo_info[walk_id][curr_node]
                    if curr_telo is None: # There is currently no telo type in the walk. 
                        curr_telo = telo_type
                        init_walk_len = len(curr_walk)
                        while True:
                            curr_node = walk[curr_ind]
                            curr_walk.append(curr_node)
                            curr_ind += 1
                            if curr_node == end_node: break
                        if init_walk_len != 0: # if there was anything before the telomeric region, include the region and chop the walk
                            new_walks.append(curr_walk.copy())
                            telo_ref[len(new_walks)-1] = {
                                'start' : None,
                                'end' : '+' if curr_telo == rep1 else '-'
                            }
                            curr_walk, curr_telo = [], None
                    elif curr_telo == telo_type: # The newly found telo type matches the current walk's telo type. Should be chopped immediately.
                        new_walks.append(curr_walk.copy())
                        telo_ref[len(new_walks)-1] = {
                            'start' : '+' if curr_telo == rep1 else '-',
                            'end' : None
                        }
                        curr_walk, curr_telo = [], telo_type
                        while True:
                            curr_node = walk[curr_ind]
                            curr_walk.append(curr_node)
                            curr_ind += 1
                            if curr_node == end_node: break
                    else: # The newly found telo type does not match the current walk's telo type. Add the telomeric region, then chop the walk.
                        while True:
                            curr_node = walk[curr_ind]
                            curr_walk.append(curr_node)
                            curr_ind += 1
                            if curr_node == end_node: 
                                new_walks.append(curr_walk.copy())
                                telo_ref[len(new_walks)-1] = {
                                    'start' : '+' if curr_telo == rep1 else '-',
                                    'end' : '+' if telo_type == rep1 else '-'
                                }
                                curr_walk, curr_telo = [], None
                                break
                else:
                    curr_walk.append(curr_node)
                    curr_ind += 1

            if curr_walk: 
                new_walks.append(curr_walk.copy())
                if curr_telo == rep1:
                    start_telo = '+'
                elif curr_telo == rep2:
                    start_telo = '-'
                else:
                    start_telo = None
                telo_ref[len(new_walks)-1] = {
                    'start' : start_telo,
                    'end' : None
                }

        # Sanity Check
        assert [item for inner in new_walks for item in inner] == [item for inner in old_walks for item in inner], "Not all nodes accounted for when chopping old walks!"

        rep1_count, rep2_count = 0, 0
        for v in telo_ref.values():
            if v['start'] == '+': rep1_count += 1
            if v['end'] == '+': rep1_count += 1
            if v['start'] == '-': rep2_count += 1
            if v['end'] == '-': rep2_count += 1
        print(f"Chopping complete! n Old Walks: {len(old_walks)}, n New Walks: {len(new_walks)}, n +ve telomeric regions: {rep1_count}, n -ve telomeric regions: {rep2_count}")

        return new_walks, telo_ref

    def add_ghosts(old_walks, paf_data, r2n, n2r, hifi_r2s, ul_r2s, n2s, old_graph, walk_valid_p, r2i):
        """
        Adds nodes and edges from the PAF and graph.

        1. Stores all nodes in the walks that are available for connection in n2n_start and n2n_end (based on walk_valid_p). 
        This is split into nodes at the start and end of walks bc incoming edges can only connect to nodes at the start of walks, and outgoing edges can only come from nodes at the end of walks.
        2. I add edges between existing walk nodes using information from PAF (although in all experiments no such edges have been found).
        3. I add nodes using information from PAF.
        4. I add nodes using information from the graph (and by proxy the GFA).
        5. I calculate the probability scores for all these new edges using GNNome's model and save them in e2s. This info is only used if decoding = 'gnnome_score'.
        """

        def is_valid_edge(rid1, rid2):
            r_info1, r_info2 = r2i[rid1], r2i[rid2]
            if r_info1['strand'] != r_info2['strand'] or r_info1['variant'] != r_info2['variant'] or r_info1['chr'] != r_info2['chr']:
                return False
        
            return (r_info1['end'] >= r_info2['start'] and r_info1['start'] > r_info2['start']) or (r_info2['end'] >= r_info1['start'] and r_info2['start'] > r_info1['start'])

        n_id = 0
        adj_list = AdjList()

        # Only the first and last walk_valid_p% of nodes in a walk can be connected. Also initialises nodes from walks
        n2n_start, n2n_end = {}, {} # n2n maps old n_id to new n_id, for the start and ends of the walks respectively
        walk_ids = [] # all n_ids that belong to walks
        nodes_in_old_walks = set()
        for walk in old_walks:
            nodes_in_old_walks.update(walk)

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

            walk_ids.append(n_id)
            n_id += 1

        print(f"Adding edges between existing nodes...")
        valid_src, valid_dst, prefix_lens, ol_lens, ol_sims, ghost_data = paf_data['ghost_edges']['valid_src'], paf_data['ghost_edges']['valid_dst'], paf_data['ghost_edges']['prefix_len'], paf_data['ghost_edges']['ol_len'], paf_data['ghost_edges']['ol_similarity'], paf_data['ghost_nodes']
        added_edges_count = 0
        for i in range(len(valid_src)):
            src, dst, prefix_len, ol_len, ol_sim = valid_src[i], valid_dst[i], prefix_lens[i], ol_lens[i], ol_sims[i]
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

        print(f"Adding ghost nodes...")
        valid_edge_count, total_edge_count = 0, 0
        n2s_ghost = {}
        ghost_data = ghost_data['hop_1'] # WE ONLY DO FOR 1-HOP FOR NOW
        added_nodes_count = 0
        for orient in ['+', '-']:
            for read_id, data in ghost_data[orient].items():
                curr_out_neighbours, curr_in_neighbours = set(), set()

                for i, out_read_id in enumerate(data['outs']):
                    out_n_id = r2n[out_read_id[0]][0] if out_read_id[1] == '+' else r2n[out_read_id[0]][1]
                    if out_n_id not in n2n_start: continue
                    curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_similarity_outs'][i], out_read_id))

                for i, in_read_id in enumerate(data['ins']):
                    in_n_id = r2n[in_read_id[0]][0] if in_read_id[1] == '+' else r2n[in_read_id[0]][1] 
                    if in_n_id not in n2n_end: continue
                    curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_similarity_ins'][i], in_read_id))

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
                    if is_valid_edge(read_id, n[4]): valid_edge_count += 1
                    total_edge_count += 1
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
                    if is_valid_edge(n[4], read_id): valid_edge_count += 1
                    total_edge_count += 1

                if orient == '+':
                    seq, _ = get_seqs(read_id, hifi_r2s, ul_r2s)
                else:
                    _, seq = get_seqs(read_id, hifi_r2s, ul_r2s)
                n2s_ghost[n_id] = seq
                n_id += 1
                added_nodes_count += 1
        print("Number of nodes added from PAF:", added_nodes_count)

        print(f"Adding nodes from old graph...")
        edges, edge_features = old_graph.edges(), old_graph.edata
        graph_data = defaultdict(lambda: defaultdict(list))
        for i in range(edges[0].shape[0]):
            src_node = edges[0][i].item()  
            dst_node = edges[1][i].item()  
            ol_len = edge_features['overlap_length'][i].item()  
            ol_sim = edge_features['overlap_similarity'][i].item()
            prefix_len = edge_features['prefix_length'][i].item() 

            if src_node not in nodes_in_old_walks:
                graph_data[src_node]['read_len'] = old_graph.ndata['read_length'][src_node]
                graph_data[src_node]['outs'].append(dst_node)
                graph_data[src_node]['ol_len_outs'].append(ol_len)
                graph_data[src_node]['ol_sim_outs'].append(ol_sim)
                graph_data[src_node]['prefix_len_outs'].append(prefix_len)

            if dst_node not in nodes_in_old_walks:
                graph_data[dst_node]['read_len'] = old_graph.ndata['read_length'][dst_node]
                graph_data[dst_node]['ins'].append(src_node)
                graph_data[dst_node]['ol_len_ins'].append(ol_len)
                graph_data[dst_node]['ol_sim_ins'].append(ol_sim)
                graph_data[dst_node]['prefix_len_ins'].append(prefix_len)

        # add to adj list where applicable
        for old_node_id, data in graph_data.items():
            curr_out_neighbours, curr_in_neighbours = set(), set()

            for i, out_n_id in enumerate(data['outs']):
                if out_n_id not in n2n_start: continue
                curr_out_neighbours.add((out_n_id, data['prefix_len_outs'][i], data['ol_len_outs'][i], data['ol_sim_outs'][i]))

            for i, in_n_id in enumerate(data['ins']):
                if in_n_id not in n2n_end: continue
                curr_in_neighbours.add((in_n_id, data['prefix_len_ins'][i], data['ol_len_ins'][i], data['ol_sim_ins'][i]))

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
                if is_valid_edge(n2r[old_node_id], n2r[n[0]]): valid_edge_count += 1
                total_edge_count += 1
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
                if is_valid_edge(n2r[n[0]], n2r[old_node_id]): valid_edge_count += 1
                total_edge_count += 1

            seq = n2s[old_node_id]
            n2s_ghost[n_id] = seq
            n_id += 1
            added_nodes_count += 1

        print("Final number of nodes:", n_id)
        if added_edges_count or added_nodes_count:
            return valid_edge_count, total_edge_count
        else:
            return None, None
    
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    postprocessing_config = config['postprocessing']
    postprocessing_config['telo_motif'] = config['genome_info'][genome]['telo_motifs']
    paths = config['genome_info'][genome]['paths']
    paths.update(config['misc']['paths'])

    print("Loading files...")
    aux = {}
    with open(paths['walks'], 'rb') as f:
        aux['walks'] = pickle.load(f)
    with open(paths['n2s'], 'rb') as f:
        aux['n2s'] = pickle.load(f)
    with open(paths['r2n'], 'rb') as f:
        aux['r2n'] = pickle.load(f)
    with open(paths['paf_processed'], 'rb') as f:
        aux['paf_data'] = pickle.load(f)
    aux['old_graph'] = dgl.load_graphs(paths['graph']+f'{genome}.dgl')[0][0]
    aux['hifi_r2s'] = Fasta(paths['ec_reads'])
    aux['ul_r2s'] = Fasta(paths['ul_reads']) if paths['ul_reads'] else None

    walks, n2s, r2n, paf_data, old_graph, hifi_r2s, ul_r2s = aux['walks'], aux['n2s'], aux['r2n'], aux['paf_data'], aux['old_graph'], aux['hifi_r2s'], aux['ul_r2s']
    rep1, rep2 = postprocessing_config['telo_motif'][0], postprocessing_config['telo_motif'][1]
    walks, telo_ref = chop_walks_seqtk(walks, n2s, old_graph, rep1, rep2, paths['seqtk'])

    print("Parsing full reads fasta...")
    r2i = defaultdict(dict)
    pattern = r"strand=(?P<strand>\S+) start=(?P<start>\d+) end=(?P<end>\d+) variant=(?P<variant>\S+) chr=(?P<chr>\S+)"
    for record in SeqIO.parse(full_reads_path, 'fasta'):
        description = record.description.split()
        match = re.search(pattern, description)
        # If a match is found, extract the values
        if match:
            r2i[record.id]['strand'] = match.group("strand")
            r2i[record.id]['start'] = match.group("start")
            r2i[record.id]['end'] = match.group("end")
            r2i[record.id]['variant'] = match.group("variant")
            r2i[record.id]['chr'] = match.group("chr")
        else:
            print("No match found!")

    n2r = {}
    for r, nodes in r2n.items():
        n2r[nodes[0]] = r
        n2r[nodes[1]] = r

    scores = {}
    for w in [0.025, 0.02, 0.015, 0.01, 0.005, 0.001]:
        valid_edge_count, total_edge_count = add_ghosts(
            old_walks=walks,
            paf_data=paf_data,
            r2n=r2n,
            n2r=n2r,
            hifi_r2s=hifi_r2s,
            ul_r2s=ul_r2s,
            n2s=n2s,
            old_graph=old_graph,
            walk_valid_p=w,
            r2i=r2i
        )
        scores[w] = (valid_edge_count, total_edge_count)
        print(f"For walk_valid_p = {w}: \nValid Edge Count: {valid_edge_count} | Total Edge Count:{total_edge_count} | Percentage: {valid_edge_count/total_edge_count:.4f}%")

def gen_kmers(name):
    print(f"=== GENERATING KMERS FOR {name} ===")
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    command = f"jellyfish count -m 100 -s 100M -t 10 -o 100mers.jf -C {config['genome_info'][name]['paths']['ec_reads']}"
    subprocess.run(command, shell=True, cwd=config['genome_info'][name]['paths']['graph'])

def parse_read(read):
    seqs = (str(read.seq), str(Seq(read.seq).reverse_complement()))
    return read.id, seqs

def parse_fasta(path):
    print(f"Parsing {path}...")

    if path.endswith('bgz'):
        if path.endswith('fasta.bgz') or path.endswith('fna.bgz') or path.endswith('fa.bgz'):
            filetype = 'fasta'
        elif path.endswith('fastq.bgz') or path.endswith('fnq.bgz') or path.endswith('fq.bgz'):
            filetype = 'fastq'
        open_func = bgzf.open
    elif path.endswith('gz'):
        if path.endswith('fasta.gz') or path.endswith('fna.gz') or path.endswith('fa.gz'):
            filetype = 'fasta'
        elif path.endswith('fastq.gz') or path.endswith('fnq.gz') or path.endswith('fq.gz'):
            filetype = 'fastq'
        open_func = gzip.open
    else:
        if path.endswith('fasta') or path.endswith('fna') or path.endswith('fa'):
            filetype = 'fasta'
        elif path.endswith('fastq') or path.endswith('fnq') or path.endswith('fq'):
            filetype = 'fastq'
        open_func = open

    data = {}
    with open_func(path, 'rt') as handle:
        rows = SeqIO.parse(handle, filetype)
        with Pool(15) as pool:
            results = pool.imap_unordered(parse_read, rows, chunksize=50)
            for id, seqs in tqdm(results, ncols=120):
                data[id] = seqs

    return data

def mers_dump(name):
    print(f"=== DUMPING KMERS FOR {name} ===")
    cmd = "jellyfish dump 21mers.jf > 21mers.fa"
    subprocess.run(cmd, shell=True, cwd=f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{name}/")

def parse_kmer(read):
    return str(read.seq), int(read.id)

def parse_kmer_fasta(path):
    print("Parsing kmer fasta...")
    data = {}
    with open(path, 'rt') as f:
        rows = SeqIO.parse(f, 'fasta')
        with Pool(40) as pool:
            results = pool.imap_unordered(parse_kmer, rows, chunksize=50)
            for kmer, freq in tqdm(results, ncols=120):
                data[kmer] = freq

    return data

if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    # for n in ['arab', 'chicken', 'mouse', 'chm13', 'maize', 'hg002_d_20x_scaf_p', 'hg002_d_20x_scaf_m', 'bonobo_d_20x_scaf_p', 'bonobo_d_20x_scaf_m', 'gorilla_d_20x_scaf_p', 'gorilla_d_20x_scaf_m']:
    #     print(f"\n=== Performing modified hifiasm decoding for {n} ===")
    #     paths = config['genome_info'][n]['paths']
    #     paths.update(config['misc']['paths'])
    #     motif = config['genome_info'][n]['telo_motifs'][0]
    #     hifiasm_decoding(paths, motif)

    # for n in ['arab', 'chicken', 'mouse', 'chm13', 'maize', 'hg002_d_20x_scaf_p', 'hg002_d_20x_scaf_m', 'bonobo_d_20x_scaf_p', 'bonobo_d_20x_scaf_m', 'gorilla_d_20x_scaf_p', 'gorilla_d_20x_scaf_m']:
    #     run_quast(n, type='res')

    # for n in ['arab_ont', 'fruitfly_ont', 'tomato_ont', 'hg005_d_ont_scaf_p', 'hg005_d_ont_scaf_m', 'hg002_d_ont_scaf_p', 'hg002_d_ont_scaf_m', 'gorilla_d_ont_20x_scaf_p', 'gorilla_d_ont_20x_scaf_m']:
    #     run_quast(n, type='res')

    # for n in ['arab', 'chicken', 'mouse', 'chm13', 'maize', 'hg002_d_20x_scaf_p', 'hg002_d_20x_scaf_m', 'bonobo_d_20x_scaf_p', 'bonobo_d_20x_scaf_m', 'gorilla_d_20x_scaf_p', 'gorilla_d_20x_scaf_m']:
    #     for c in range(5):
    #         print("COUNT:", c)
    #         analyse_t2t(n, type='baseline')
    #         count_t2t(n, type='baseline')
    #         analyse_t2t(n, type='res')
    #         count_t2t(n, type='res')
    #         print("\n")

    # for n in ['arab_ont', 'tomato_ont', 'hg005_d_ont_scaf_p', 'hg005_d_ont_scaf_m', 'hg002_d_ont_scaf_p', 'hg002_d_ont_scaf_m', 'gorilla_d_ont_20x_scaf_p', 'gorilla_d_ont_20x_scaf_m']:
    #     for c in range(5):
    #         print("COUNT:", c)
    #         analyse_t2t(n, type='baseline')
    #         count_t2t(n, type='baseline')
    #         analyse_t2t(n, type='res')
    #         count_t2t(n, type='res')
    #         print("\n")

    # for n in ['gorilla_d_ont_20x_scaf']:
    #     ec_path = f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{n}/{n}.ec.fa'
    #     data = parse_fasta(ec_path)
    #     with open(f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{n}/r2s.pkl', "wb") as p:
    #         pickle.dump(data, p)

    for n in ['arab', 'chicken', 'mouse', 'chm13', 'maize', 'chm13', 'hg002_d_20x_scaf', 'bonobo_d_20x_scaf', 'gorilla_d_20x_scaf', 'arab_ont', 'fruitfly_ont', 'tomato_ont', 'hg005_d_ont_scaf', 'hg002_d_ont_scaf', 'gorilla_d_ont_20x_scaf']:
        print("running for", n)
        path = f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{n}/21mers.fa"
        res = parse_kmer_fasta(path)
        with open(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/GAP/hifiasm/{n}/21mers.pkl", "wb") as p:
            pickle.dump(res, p)
