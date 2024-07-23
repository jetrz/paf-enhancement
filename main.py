import argparse
import pickle
import torch
import warnings

from decision_node_util import add_decision_nodes
from gfa_util import only_from_gfa
from fasta_util import parse_fasta, add_train_labels
from paf_util import parse_paf, enhance_with_paf, enhance_with_paf_2, check_duplicate_edges

def edge_exists(edge_index, src_id, dst_id):
    # Convert edge_index to a 2D numpy array for easier processing
    edge_pairs = edge_index.t().numpy()
    
    # Check if the (src_id, dst_id) pair exists in the array of edge pairs
    return any((edge[0] == src_id and edge[1] == dst_id) for edge in edge_pairs)

if __name__ == "__main__":
    '''
    Currently, this creates 4 files:
    - graphs/mode/<name>.pt - Saved graph
    - pkl/<name>_paf_data.pkl - Saved paf data pickle, if mode != 'default'
    - pkl/<name>_fasta_data.pkl - Saved fasta data pickle, if mode != 'default'
    - pkl/<name>_train_fasta_data.pkl - Saved training fasta data pickle, if mode != 'default' and test is False
    '''
    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--mode", type=str, default='default', help="default, ghost1, ghost1-1, ghost1-2, ghost2, ghost2-1. See README.md for details.")
    parser.add_argument("--pickle", type=bool, default=False, help="whether or not to use pkl file to load paf data")
    parser.add_argument("--test", type=bool, default=False, help="generate graphs for test or train")

    args = parser.parse_args()
    mode = args.mode
    use_pickle = args.pickle
    is_test = args.test
    warnings.filterwarnings("ignore")

    if is_test:
        ref = {
            'chm13' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/PacBio_HiFi/SRR11292120_3_subreads.fastq.gz', 'gfa' : '/mnt/sod2-project/csb4/wgs/martin/real_haploid_datasets/real_chm13/gfa_graphs/full_0.gfa'},
            'arab' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/arabidopsis_new/PacBio_HiFi/CRR302668_p0.22.fastq.gz', 'gfa' : '/mnt/sod2-project/csb4/wgs/martin/real_haploid_datasets/arabidopsis/gfa_graphs/full_0.gfa'},
            'mouse' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/mus_musculus/SRR11606870.fastq', 'gfa' : '/mnt/sod2-project/csb4/wgs/martin/real_haploid_datasets/mmusculus/gfa_graphs/full_0.gfa'},
            'chicken' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/gallus_gallus/HiFi/mat_0.5_30x.fastq.gz', 'gfa' : '/mnt/sod2-project/csb4/wgs/martin/real_haploid_datasets/chicken/gfa_graphs/full_0.gfa'}
        }

        for i in ['arab', 'chicken', 'chm13', 'mouse']:
            gfa_path = ref[i]['gfa']
            paf_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{i}.ovlp.paf"
            annotated_fasta_path = ref[i]['fasta']
            get_similarities = True

            print("Starting run for:", i)

            ##### Creates initial graph from GFA. #####
            g, aux = only_from_gfa(gfa_path=gfa_path, get_similarities=get_similarities)
            print('\ng before enhance:', g, '\n')

            # for i in [(7769, 33315), (33314, 7768), (8257, 64230), (64231, 8256)]:
            #     print(f"edge exists: {i[0]} -> {i[1]}:", edge_exists(g.edge_index, i[0], i[1]))

            ##### Parsing FASTA file, because read start and end data is required for graph enhancement via PAF. Train data is also retrieved for use later potentially. #####
            if use_pickle:
                with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{i}_fasta_data.pkl', 'rb') as f:
                    aux['annotated_fasta_data'] = pickle.load(f)
            else:
                aux['annotated_fasta_data'], _ = parse_fasta(annotated_fasta_path, i, training=False)

            if mode == 'ghost2':
                if use_pickle:
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{i}_paf_data.pkl', 'rb') as f:
                        aux['paf_data'] = pickle.load(f)
                else:
                    aux['paf_data'] = parse_paf(paf_path, aux, i)

                aux['node_attrs'].append('is_real_node')
                aux['edge_attrs'].append('is_real_edge')
                g, aux = enhance_with_paf_2(g, aux, get_similarities=get_similarities, add_features=True)
                print("\ng after ghost enhance:", g, '\n')

            print("check duplicate edges for g:")
            check_duplicate_edges(g)

            torch.save(g, f'static/graphs/{mode}/{i}.pt')
            print('Done!\n')
    else:
        ref = {}
        for i in [1,3,5,9,12,18]:
            ref[i] = [i for i in range(15)]
        for i in [11,16,17,19,20]:
            ref[i] = [i for i in range(5)]

        # for chr in [1,3,5,9,12,18,11,16,17,19,20]:
        for chr in [9]:
            for i in ref[chr]:
                genome = f'chr{chr}_M_{i}'        
                gfa_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{genome}_asm.bp.raw.r_utg.gfa"
                paf_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{genome}_asm.ovlp.paf"
                annotated_fasta_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{genome}.fasta"
                get_similarities = True

                print("Starting run for genome:", genome)

                ##### Creates initial graph from GFA. #####
                g, aux = only_from_gfa(gfa_path=gfa_path, get_similarities=get_similarities)
                print('\ng before enhance:', g, '\n')

                ##### Parsing FASTA file, because read start and end data is required for graph enhancement via PAF. Train data is also retrieved for use later potentially. #####
                if use_pickle:
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_fasta_data.pkl', 'rb') as f:
                        aux['annotated_fasta_data'] = pickle.load(f)
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_train_fasta_data.pkl', 'rb') as f:
                        aux['train_data'] = pickle.load(f)
                else:
                    aux['annotated_fasta_data'], aux['train_data'] = parse_fasta(annotated_fasta_path, genome, training=True)

                if mode.startswith('ghost'):
                    ###################### Parsing PAF file, and enhancing graph. This does two things: ######################
                    if use_pickle:
                        with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_paf_data.pkl', 'rb') as f:
                            aux['paf_data'] = pickle.load(f)
                    else:
                        aux['paf_data'] = parse_paf(paf_path, aux, genome)

                    if mode.startswith('ghost1'):
                        add_edges, add_node_features = True, True
                        if mode == 'ghost1-1':
                            add_node_features = False
                        elif mode == 'ghost1-2':
                            add_edges = False
                            
                        if mode != 'ghost1-1': aux['node_attrs'].extend(['ghost_n_outs', 'ghost_ol_len_outs', 'ghost_ol_sim_outs', 'ghost_n_ins', 'ghost_ol_len_ins', 'ghost_ol_sim_ins'])

                        g = enhance_with_paf(g, aux, get_similarities=get_similarities, add_edges=add_edges, add_node_features=add_node_features)
                    elif mode.startswith('ghost2'):
                        add_features = True

                        if mode == 'ghost2-1':
                            add_features = False

                        if mode != 'ghost2-1':
                            aux['node_attrs'].append('is_real_node')
                            aux['edge_attrs'].append('is_real_edge')
                        
                        # Note that enhance_with_paf2 adds new nodes and edges. 
                        # However, in aux, it only updates 'node_to_read' and not 'read_to_node'/'read_seqs' due to differences in implementation.
                        g, aux = enhance_with_paf_2(g, aux, get_similarities=get_similarities, add_features=add_features)
                    else:
                        raise ValueError("Unrecognised mode!")
                    
                    print("\ng after ghost enhance:", g, '\n')

                # check_duplicate_edges(g)

                ##### Adds training labels. #####
                g, aux = add_train_labels(g, aux)

                ##### Adds decision nodes. #####
                params = { "deadends" : {} }
                g, aux = add_decision_nodes(g, aux, params)

                print("check duplicate edges for g:")
                check_duplicate_edges(g)

                print('\ng after enhance:', g, '\n')
                torch.save(g, f'static/graphs/{mode}/{genome}.pt')

                print('Done!\n')
