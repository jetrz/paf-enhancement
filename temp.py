import dgl, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def analyse(name):
    print(f"\n=== ANALYSING FOR {name} ===\n")
    g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/{name}.dgl")[0][0]
    # g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/chr1_M_0.dgl")[0][0]
    print("graph loaded! number of edges:", len(g.edata['E_ID']))

    # Extract edge data from the graph
    edge_data = g.edata

    # Filter edges where edge_hop == 1
    edge_hop_1_mask = edge_data['edge_hop'] == 1
    gt_bin_0_mask = edge_data['gt_bin'] == 0
    combined_mask = edge_hop_1_mask & gt_bin_0_mask

    # Compute OL percentiles (50th, 75th, and 90th)
    filtered_overlap_length_gt_bin_0 = edge_data['overlap_length'][combined_mask].numpy()
    percentiles = [50, 75, 90]
    percentile_values = np.percentile(filtered_overlap_length_gt_bin_0, percentiles)
    print(f"OL 50th percentile: {percentile_values[0]}")
    print(f"OL 75th percentile: {percentile_values[1]}")
    print(f"OL 90th percentile: {percentile_values[2]}")

    # Get filtered features
    filtered_gt_bin = edge_data['gt_bin'][edge_hop_1_mask].numpy()
    filtered_overlap_length = edge_data['overlap_length'][edge_hop_1_mask].numpy()
    filtered_overlap_similarity = edge_data['overlap_similarity'][edge_hop_1_mask].numpy()

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
    plt.savefig(f"graphs/train/{name}_oll_ols_corr.png")
    plt.clf()

    return percentile_values[2]

ref = {}
for i in [1,3,5,9,12,18]:
    ref[i] = [j for j in range(15)]
for i in [11,16,17,19,20]:
    ref[i] = [j for j in range(5)]

res = []
for chr in ref.keys():
    selected = random.sample(ref[chr], 5)
    for i in selected:
        res.append(analyse(f"chr{chr}_M_{i}"))
print(f"Final Average 90th Percentile Value: {sum(res)/len(res)}")