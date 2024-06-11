import gzip, re, sys, os
from collections import Counter
from copy import deepcopy
from datetime import datetime
import pickle

from Bio import SeqIO
from Bio.Seq import Seq
import torch
import edlib
from tqdm import tqdm
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from algorithms import process_graph, process_graph_combo
from parse_fasta import parse_fasta
from paf_util import parse_paf, enhance_with_paf, analyse, analyse2

def only_from_gfa(gfa_path, training=False, reads_path=None, get_similarities=False, is_hifiasm=True):
    if training:
        print(f'Parsing reads file...')
        if reads_path is not None:
            if reads_path.endswith('gz'):
                if reads_path.endswith('fasta.gz') or reads_path.endswith('fna.gz') or reads_path.endswith('fa.gz'):
                    filetype = 'fasta'
                elif reads_path.endswith('fastq.gz') or reads_path.endswith('fnq.gz') or reads_path.endswith('fq.gz'):
                    filetype = 'fastq'
                with gzip.open(reads_path, 'rt') as handle:
                    read_headers = {read.id: read.description for read in SeqIO.parse(handle, filetype)}
            else:
                if reads_path.endswith('fasta') or reads_path.endswith('fna') or reads_path.endswith('fa'):
                    filetype = 'fasta'
                elif reads_path.endswith('fastq') or reads_path.endswith('fnq') or reads_path.endswith('fq'):
                    filetype = 'fastq'
                read_headers = {read.id: read.description for read in SeqIO.parse(reads_path, filetype)}
        else:
            print('You need to pass the reads_path with annotations')
            exit(1)

    # Loop through GFA. Assume first N lines start with "S"
    time_start = datetime.now()
    print("Reading GFA...")
    with open(gfa_path) as f:
        rows = f.readlines()

    n_rows = len(rows)    
    N_ID, E_ID = 0, 0

    read_to_node, node_to_read = {}, {}
    read_to_node2 = {}
    read_lengths, read_seqs = {}, {}  # Obtained from the GFA
    read_strands, read_starts, read_ends, read_chrs = {}, {}, {}, {}  # Obtained from the FASTA/Q headers
    edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}
    edge_index = [[],[]]

    r_ind = 0
    while r_ind < n_rows:
        row = rows[r_ind].strip().split()
        tag = row.pop(0)

        if tag == 'S':
            if len(row) == 4: # Hifiasm
                s_id, seq, length, count = row
            elif len(row) == 3: # Raven
                s_id, seq, length = row
            else:
                raise Exception("Unknown GFA format!")

            real_id, virt_id = N_ID, N_ID + 1 # 1+, 1-
            N_ID += 2
            read_to_node[s_id] = (real_id, virt_id)
            node_to_read[real_id] = s_id
            node_to_read[virt_id] = s_id

            # DID NOT ADD NODES TO GRAPH YET (LINE 179, 180 IN DGL FILE)

            no_seqs_flag = bool(seq == "*")
            seq = Seq(seq)
            read_seqs[real_id] = str(seq)
            read_seqs[virt_id] = str(seq.reverse_complement())

            length = int(length[5:])
            read_lengths[real_id] = length
            read_lengths[virt_id] = length

            if s_id.startswith('utg'):
                # The issue here is that in some cases, one unitig can consist of more than one read
                # So this is the adapted version of the code that supports that
                # The only things of importance here are read_to_node2 dict (not overly used)
                # And id variable which I use for obtaining positions during training (for the labels)
                # I don't use it for anything else, which is good
                s_ids = []
                while rows[r_ind+1][0] == 'A':
                    r_ind += 1
                    row = rows[r_ind].strip().split()
                    read_orientation, utg_to_read = row[3], row[4]
                    s_ids.append((utg_to_read, read_orientation))
                    read_to_node2[utg_to_read] = (real_id, virt_id)

                s_id = s_ids
                node_to_read[real_id] = s_id
                node_to_read[virt_id] = s_id

            if training:
                if type(s_id) != list:
                    description = read_headers[s_id]
                    # desc_id, strand, start, end = description.split()
                    strand = re.findall(r'strand=(\+|\-)', description)[0]
                    strand = 1 if strand == '+' else -1
                    start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                    end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                    chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                else:
                    strands, starts, ends, chromosomes = [], [], [], []
                    for id_r, id_o in s_id:
                        description = read_headers[id_r]
                        strand_fasta = re.findall(r'strand=(\+|\-)', description)[0]
                        strand_fasta = 1 if strand_fasta == '+' else -1
                        strand_gfa = 1 if id_o == '+' else -1
                        strand = strand_fasta * strand_gfa

                        strands.append(strand)
                        start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                        starts.append(start)
                        end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                        ends.append(end)
                        chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                        chromosomes.append(chromosome)

                    # What if they come from different strands but are all merged in a single unitig?
                    # Or even worse, different chromosomes? How do you handle that?
                    # I don't think you can. It's an error in the graph
                    strand = 1 if sum(strands) >= 0 else -1
                    start = min(starts)
                    end = max(ends)
                    chromosome = Counter(chromosomes).most_common()[0][0]

                read_strands[real_id], read_strands[virt_id] = strand, -strand
                read_starts[real_id] = read_starts[virt_id] = start
                read_ends[real_id] = read_ends[virt_id] = end
                read_chrs[real_id] = read_chrs[virt_id] = chromosome
        elif tag == 'L':
            if len(row) == 5: # raven, normal GFA 1 standard
                s_id1, orient1, s_id2, orient2, cigar = row
            elif len(row) == 6: # Hifiasm GFA
                s_id1, orient1, s_id2, orient2, cigar, _ = row
                s_id1 = re.findall(r'(.*):\d-\d*', s_id1)[0]
                s_id2 = re.findall(r'(.*):\d-\d*', s_id2)[0]
            elif len(row) == 7: # Hifiasm GFA newer
                s_id1, orient1, s_id2, orient2, cigar, _, _ = row
            else:
                raise Exception("Unknown GFA format!")
            
            if orient1 == '+' and orient2 == '+':
                src_real = read_to_node[s_id1][0]
                dst_real = read_to_node[s_id2][0]
                src_virt = read_to_node[s_id2][1]
                dst_virt = read_to_node[s_id1][1]
            elif orient1 == '+' and orient2 == '-':
                src_real = read_to_node[s_id1][0]
                dst_real = read_to_node[s_id2][1]
                src_virt = read_to_node[s_id2][0]
                dst_virt = read_to_node[s_id1][1]
            elif orient1 == '-' and orient2 == '+':
                src_real = read_to_node[s_id1][1]
                dst_real = read_to_node[s_id2][0]
                src_virt = read_to_node[s_id2][1]
                dst_virt = read_to_node[s_id1][0]
            elif orient1 == '-' and orient2 == '-':
                src_real = read_to_node[s_id1][1]
                dst_real = read_to_node[s_id2][1]
                src_virt = read_to_node[s_id2][0]
                dst_virt = read_to_node[s_id1][0]  
            else:
                raise Exception("Unknown GFA format!")
            
            if is_hifiasm: # Don't need to manually add reverse complement edge
                edge_index[0].append(src_real)
                edge_index[1].append(dst_real)
                edge_ids[(src_real, dst_real)] = E_ID
                E_ID += 1
            else:
                edge_index[0].extend([src_real, src_virt])
                edge_index[1].extend([dst_real, dst_virt])
                edge_ids[(src_real, dst_real)] = E_ID
                edge_ids[(src_virt, dst_virt)] = E_ID + 1
                E_ID += 2

            # -----------------------------------------------------------------------------------
            # This enforces similarity between the edge and its "virtual pair"
            # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
            # When parsing CSV that was not necessarily so:
            # Sometimes reads would be slightly differently aligned from their RC pairs
            # Thus resulting in different overlap lengths
            # -----------------------------------------------------------------------------------

            try:
                ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
            except ValueError:
                print('Cannot convert CIGAR string into overlap length!')
                raise ValueError
            
            overlap_lengths[(src_real, dst_real)] = ol_length
            overlap_lengths[(src_virt, dst_virt)] = ol_length

            prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
            prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length

        r_ind += 1
            
    if no_seqs_flag:
        print("No sequences in GFA detected. Reading FASTA/Q file...")
        if reads_path.endswith('gz'):
            if reads_path.endswith('fasta.gz') or reads_path.endswith('fna.gz') or reads_path.endswith('fa.gz'):
                filetype = 'fasta'
            elif reads_path.endswith('fastq.gz') or reads_path.endswith('fnq.gz') or reads_path.endswith('fq.gz'):
                filetype = 'fastq'
            with gzip.open(reads_path, 'rt') as handle:
                fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(handle, filetype)}
        else:
            if reads_path.endswith('fasta') or reads_path.endswith('fna') or reads_path.endswith('fa'):
                filetype = 'fasta'
            elif reads_path.endswith('fastq') or reads_path.endswith('fnq') or reads_path.endswith('fq'):
                filetype = 'fastq'
            fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}

        print(f'Sequences successfully loaded!')
        # fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}
        for node_id in tqdm(read_seqs.keys(), ncols=120):
            read_id = node_to_read[node_id]
            seq = fastaq_seqs[read_id]
            read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
        print(f'Loaded DNA sequences!')

    elapsed = (datetime.now() - time_start).seconds
    print(f"Run Time: {elapsed}s, Creating graph...")

    if get_similarities:
        print("Calculating similarities...")
        overlap_similarities = calculate_similarities(edge_ids, read_seqs, overlap_lengths)

    g = Data(N_ID=torch.tensor([i for i in range(N_ID)]), E_ID=torch.tensor([i for i in range(E_ID)]), edge_index=torch.tensor(edge_index))
    aux = { 'read_lengths_dict' : read_lengths, 'prefix_lengths_dict' : prefix_lengths, 'overlap_lengths_dict' : overlap_lengths }

    node_attrs, edge_attrs = ['read_length'], ['prefix_length', 'overlap_length']
    # Only convert to list right before creating graph data
    read_lengths_list = [read_lengths[i] for i in range(N_ID)]
    prefix_lengths_list, overlap_lengths_list = [0]*E_ID, [0]*E_ID
    for k, e_id in edge_ids.items():
        prefix_lengths_list[e_id] = prefix_lengths[k]
        overlap_lengths_list[e_id] = overlap_lengths[k]
    g['read_length'] = torch.tensor(read_lengths_list)
    g['prefix_length'] = torch.tensor(prefix_lengths_list)
    g['overlap_length'] = torch.tensor(overlap_lengths_list)

    if training:
        print("Adding training labels...")
        node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_chr'])

        # Only convert to list right before creating graph data
        read_strand_list, read_start_list, read_end_list, read_chr_list = [read_strands[i] for i in range(N_ID)], [read_starts[i] for i in range(N_ID)], [read_ends[i] for i in range(N_ID)], [read_chrs[i] for i in range(N_ID)]
        g['read_strand'] = torch.tensor(read_strand_list)
        g['read_start'] = torch.tensor(read_start_list)
        g['read_end'] = torch.tensor(read_end_list)
        g['read_chr'] = torch.tensor(read_chr_list)
        aux['read_strand_dict'] = read_strands
        aux['read_start_dict'] = read_starts
        aux['read_end_dict'] = read_ends
        aux['read_chr_dict'] = read_chrs

        nx_g = to_networkx(data=g, node_attrs=node_attrs, edge_attrs=edge_attrs)

        unique_chrs = set(read_chrs.values())
        if len(unique_chrs) == 1:
            ms_pos, labels = process_graph(nx_g)
        else:
            ms_pos, labels = process_graph_combo(nx_g)
        nx.set_edge_attributes(nx_g, labels, 'y')
        labels = torch.tensor([data['y'] for _, _, data in nx_g.edges(data=True)])
        
        g['y'] = labels
        edge_attrs.append('y')

    if get_similarities:
        overlap_similarities_list = [0]*E_ID
        for k, e_id in edge_ids.items():
            overlap_similarities_list[e_id] = overlap_similarities[k]
        g['overlap_similarity'] = torch.tensor(overlap_similarities_list)
        aux['overlap_similarities_dict'] = overlap_similarities
        edge_attrs.append('overlap_similarity')

    # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
    if len(read_to_node2) != 0:
        read_to_node = read_to_node2

    aux = {
        'read_to_node' : read_to_node,
        'read_seqs' : read_seqs
    }

    return g, aux

def calculate_similarities(edge_ids, read_seqs, overlap_lengths):
    # Make sure that read_seqs is a dict of string, not Bio.Seq objects!
    overlap_similarities = {}
    for src, dst in tqdm(edge_ids.keys(), ncols=120):
        ol_length = overlap_lengths[(src, dst)]
        read_src = read_seqs[src]
        read_dst = read_seqs[dst]
        edit_distance = edlib.align(read_src[-ol_length:], read_dst[:ol_length])['editDistance']
        overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length
    return overlap_similarities

def reverse_graph(g):
    """
    Reverses a PyG graph by swapping rows in edge_index.
    """
    g_copy = deepcopy(g.detach())
    rev_e_index = g_copy.edge_index[[1,0], :]
    g_copy.edge_index = rev_e_index
    return g_copy

if __name__ == "__main__":
    # genome = 'arab'

    for chr in [5, 9, 12, 18]:
        for i in range(10, 15):
            genome = f'chr{chr}_M_{i}'        
            gfa_path = f"datasets/{genome}/{genome}_asm.bp.raw.r_utg.gfa"
            paf_path = f"datasets/{genome}/{genome}_asm.ovlp.paf"
            annotated_fasta_path = f"datasets/{genome}/{genome}.fasta"
            get_similarities = True

            print("Starting run for genome:", genome)

            g, aux = only_from_gfa(gfa_path=gfa_path, get_similarities=get_similarities)
            print('g before enhance:', g)

            aux['annotated_fasta_data'] = parse_fasta(annotated_fasta_path)

            aux['paf_data'] = parse_paf(paf_path, aux, genome)
            # with open(f'static/pkl/{genome}_paf_data.pkl', 'rb') as f:
            #     aux['paf_data'] = pickle.load(f)

            analyse(aux['paf_data'], genome)

            g = enhance_with_paf(g, aux, genome, get_similarities=get_similarities)
            print('g after enhance:', g)

            analyse2(g, genome)

            print('Done!\n')