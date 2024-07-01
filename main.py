import argparse
import pickle
import torch
import warnings

from decision_node_util import add_decision_nodes
from gfa_util import only_from_gfa
from fasta_util import parse_fasta, add_train_labels
from paf_util import parse_paf, enhance_with_paf, enhance_with_paf_2, analyse, analyse2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--mode", type=str, default='default', help="default or ghost")
    parser.add_argument("--pickle", type=bool, default=False, help="whether or not to use pkl file to load paf data")

    args = parser.parse_args()
    mode = args.mode
    use_pickle = args.pickle
    
    warnings.filterwarnings("ignore")

    ref = {}
    for i in [1,3,5,9,12,18]:
        ref[i] = [i for i in range(15)]
    for i in [11,16,17,19,20]:
        ref[i] = [i for i in range(5)]

    for chr in [11,16,17,19,20]:
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
            aux['annotated_fasta_data'], aux['train_data'] = parse_fasta(annotated_fasta_path, training=True)

            if mode == 'ghost1' or mode == 'ghost2':
                ###################### Parsing PAF file, and enhancing graph. This does two things: ######################
                ##### 1. Any valid edges in paf file that are between two existing nodes are added to the graph, and #####
                ######################## 2. Ghost node node features are added to existing nodes. ########################
                if use_pickle:
                    with open(f'static/pkl/{genome}_paf_data.pkl', 'rb') as f:
                        aux['paf_data'] = pickle.load(f)
                else:
                    aux['paf_data'] = parse_paf(paf_path, aux, genome)

                if mode == 'ghost1':
                    aux['node_attrs'].extend(['ghost_n_outs', 'ghost_ol_len_outs', 'ghost_ol_sim_outs', 'ghost_n_ins', 'ghost_ol_len_ins', 'ghost_ol_sim_ins'])
                    g = enhance_with_paf(g, aux, get_similarities=get_similarities)
                else:
                    # Note that enhance_with_paf2 adds new nodes and edges. 
                    # However, in aux, it only updates 'node_to_read' and not 'read_to_node'/'read_seqs' due to differences in implementation.
                    g, aux = enhance_with_paf_2(g, aux, get_similarities=get_similarities)
                
                print("\ng after ghost enhance:", g, '\n')


            ##### Adds training labels. #####
            g, aux = add_train_labels(g, aux)

            ##### Adds decision nodes. #####
            params = { "deadends" : {} }
            g, aux = add_decision_nodes(g, aux, params)

            print('\ng after enhance:', g, '\n')
            torch.save(g, f'static/graphs/{mode}/{mode}_{genome}.pt')

            print('Done!\n')