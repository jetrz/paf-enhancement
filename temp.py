import dgl, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from tqdm import tqdm

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
                c_seq = seq[:cutoff] if j == 'start' else seq[-cutoff//10:]
                curr_res[j]['len'] = len(c_seq)
                rep1_count, rep2_count = count_seq(c_seq, rep1), count_seq(c_seq, rep2)
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

for name in ['arab', 'chicken', 'mouse', 'chm13']:
    analyse_telomere(name)