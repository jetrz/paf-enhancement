import argparse
import pickle
import torch
import warnings
import dgl

from decision_node_util import add_decision_nodes
from gfa_util import only_from_gfa
from fasta_util import parse_fasta, add_train_labels
from paf_util import parse_paf, enhance_with_paf_2, check_duplicate_edges
from misc_util import pyg_to_dgl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--mode", type=str, default='default', help="ghost-<hop>. See README.md for details.")
    parser.add_argument("--test", type=lambda x: (str(x).lower() == 'true'), default=False, help="generate graphs for test or train")
    parser.add_argument("--add_inner_edges", type=lambda x: (str(x).lower() == 'true'), default=True, help="add inner edges for ghost mode")

    parser.add_argument("--pickle_gfa", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether or not to use pkl file to load gfa data")
    parser.add_argument("--pickle_fasta", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether or not to use pkl file to load fasta data")
    parser.add_argument("--pickle_paf", type=lambda x: (str(x).lower() == 'true'), default=False, help="whether or not to use pkl file to load paf data")

    args = parser.parse_args()
    mode = args.mode
    is_test = args.test
    add_inner_edges = args.add_inner_edges

    pickle_gfa = args.pickle_gfa
    pickle_fasta = args.pickle_fasta
    pickle_paf = args.pickle_paf
    warnings.filterwarnings("ignore")

    if is_test:
        ref = {
            'chm13' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/CHM13/PacBio_HiFi/SRR11292120_3_subreads.fastq.gz'},
            'arab' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/arabidopsis_new/PacBio_HiFi/CRR302668_p0.22.fastq.gz'},
            'mouse' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/mus_musculus/SRR11606870.fastq'},
            'chicken' : {'fasta' : '/mnt/sod2-project/csb4/wgs/lovro/sequencing_data/gallus_gallus/HiFi/mat_0.5_30x.fastq.gz'}
        }

        for i in ['chicken', 'arab', 'chm13', 'mouse']:
            gfa_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{i}.bp.raw.r_utg.gfa"
            paf_path = f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/{i}.ovlp.paf"
            annotated_fasta_path = ref[i]['fasta']
            get_similarities = True

            print("Starting run for:", i)

            ##### Creates initial graph from GFA. #####
            if pickle_gfa:
                g = torch.load('../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/temp/gfa_g.pt')
                with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/temp/gfa_aux.pkl', 'rb') as f:
                    aux = pickle.load(f)
            else:
                g, aux = only_from_gfa(gfa_path=gfa_path, get_similarities=get_similarities)
                torch.save(g, f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/temp/gfa_g.pt')
                with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/temp/gfa_aux.pkl", "wb") as p:
                    pickle.dump(aux, p)
            print('\ng before enhance:', g, '\n')


            ##### Parsing FASTA file, because read start and end data is required for graph enhancement via PAF. Train data is also retrieved for use later potentially. #####
            if pickle_fasta:
                with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{i}_fasta_data.pkl', 'rb') as f:
                    aux['annotated_fasta_data'] = pickle.load(f)
            else:
                aux['annotated_fasta_data'], _ = parse_fasta(annotated_fasta_path, i, training=False)

            if mode.startswith('ghost'):
                mode_split = mode.split('-')
                hop = int(mode_split[1])

                if pickle_paf:
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{i}_paf_data.pkl', 'rb') as f:
                        aux['paf_data'] = pickle.load(f)
                else:
                    aux['paf_data'] = parse_paf(paf_path, aux, i)

                aux['node_attrs'].append('node_hop')
                aux['edge_attrs'].append('edge_hop')

                g, aux = enhance_with_paf_2(g, aux, get_similarities=get_similarities, hop=hop, add_inner_edges=add_inner_edges)
                print("\ng after ghost enhance:", g, '\n')

            # print("check duplicate edges for g:")
            # check_duplicate_edges(g)

            # with open(f"../scratch/{mode}_{i}_n2s.pkl", "wb") as p:
            with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{mode}_{i}_n2s.pkl", "wb") as p:
                pickle.dump(aux['read_seqs'], p)
            with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{mode}_{i}_r2n.pkl", "wb") as p:
                pickle.dump(aux['read_to_node'], p)

            graph_path =  f'/mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/{mode}/'
            if not add_inner_edges: graph_path += 'no_inner_edges/'
            torch.save(g, f'{graph_path}{i}.pt')
            dgl_g = pyg_to_dgl(g, False, mode)
            dgl.save_graphs(f'{graph_path}{i}.dgl', [dgl_g])

            print('Done!\n')
    else:
        ref = {}
        for i in [1,3,5,9,12,18]:
            ref[i] = [i for i in range(15)]
        for i in [11,16,17,19,20]:
            ref[i] = [i for i in range(5)]
            
        for chr in [1,3,5,9,12,18,11,16,17,19,20]:
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
                if pickle_fasta:
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_fasta_data.pkl', 'rb') as f:
                        aux['annotated_fasta_data'] = pickle.load(f)
                    with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_train_fasta_data.pkl', 'rb') as f:
                        aux['train_data'] = pickle.load(f)
                else:
                    aux['annotated_fasta_data'], aux['train_data'] = parse_fasta(annotated_fasta_path, genome, training=True)

                if mode.startswith('ghost'):
                    mode_split = mode.split('-')
                    hop = int(mode_split[1])

                    if pickle_paf:
                        with open(f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{genome}_paf_data.pkl', 'rb') as f:
                            aux['paf_data'] = pickle.load(f)
                    else:
                        aux['paf_data'] = parse_paf(paf_path, aux, genome)

                    aux['node_attrs'].append('node_hop')
                    aux['edge_attrs'].append('edge_hop')
                        
                    g, aux = enhance_with_paf_2(g, aux, get_similarities=get_similarities, hop=hop, add_inner_edges=add_inner_edges)
                    print("\ng after ghost enhance:", g, '\n')

                ##### Adds training labels. #####
                g, aux = add_train_labels(g, aux)

                ##### Adds decision nodes. #####
                params = { "deadends" : {} }
                g, aux = add_decision_nodes(g, aux, params)

                # print("check duplicate edges for g:")
                # check_duplicate_edges(g)

                with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{mode}_{genome}_n2s.pkl", "wb") as p:
                    pickle.dump(aux['read_seqs'], p)
                with open(f"../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/pkl/{mode}_{genome}_r2n.pkl", "wb") as p:
                    pickle.dump(aux['read_to_node'], p)

                print('\ng after enhance:', g, '\n')
                graph_path =  f'../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/graphs/{mode}/'
                if not add_inner_edges: graph_path += 'no_inner_edges/'
                torch.save(g, f'{graph_path}{genome}.pt')
                dgl_g = pyg_to_dgl(g, True, mode)
                dgl.save_graphs(f'{graph_path}{genome}.dgl', [dgl_g])

                print('Done!\n')
