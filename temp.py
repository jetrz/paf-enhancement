import dgl, random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyse(name):
    print(f"\n=== ANALYSING FOR {name} ===\n")
    g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/{name}.dgl")[0][0]
    # g = dgl.load_graphs(f"/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/ghost-1/chr1_M_0.dgl")[0][0]
    print("graph loaded!")

    # Extract edge data from the graph
    edge_data = g.edata

    # Filter edges where edge_hop == 1
    edge_hop_1_mask = edge_data['edge_hop'] == 1

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
    sns.stripplot(x='overlap_length', y='gt_bin', data=df, jitter=0.1, size=4)

    # Plot for overlap_similarity vs gt_bin
    plt.subplot(1, 2, 2)
    sns.stripplot(x='overlap_similarity', y='gt_bin', data=df, jitter=0.1, size=4)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"graphs/train/{name}_oll_ols_corr.png")

ref = {}
for i in [1,3,5,9,12,18]:
    ref[i] = [j for j in range(15)]
for i in [11,16,17,19,20]:
    ref[i] = [j for j in range(5)]

for chr in ref.keys():
    selected = random.sample(ref[chr], 2)
    for i in selected:
        analyse(f"chr{chr}_M_{i}")