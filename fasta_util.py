from Bio import SeqIO
from Bio.Seq import Seq
from collections import Counter
import gzip, torch, re
import networkx as nx
from tqdm import tqdm
from torch_geometric.utils import to_networkx

from algorithms import process_graph, process_graph_combo

def parse_fasta(path, training=False):
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

    data, train_data = {}, {}
    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as handle:
        for read in tqdm(SeqIO.parse(handle, filetype), ncols=120):
            description = read.description.split()
            id = description[0]
            data[id] = (read.seq, str(Seq(read.seq).reverse_complement()))

            if training:
                train_data[id] = read.description

    return data, train_data

def add_train_labels(g, aux):
    print("Adding training labels...")
    read_strands, read_starts, read_ends, read_chrs = {}, {}, {}, {}
    train_data, n2r = aux['train_data'], aux['node_to_read']
    if not train_data:
        print("Did you forget to set the train parameter correctly?")
        raise(RuntimeError)
    
    n_id, n_nodes = 0, g.N_ID.size()[0]
    while n_id < n_nodes-1:
        real_id, virt_id = n_id, n_id+1
        n_id += 2
        s_id = n2r[real_id]

        if type(s_id) != list:
            description = train_data[s_id]
            # desc_id, strand, start, end = description.split()
            strand = re.findall(r'strand=(\+|\-)', description)[0]
            strand = 1 if strand == '+' else -1
            start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
            end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
            chromosome = int(re.findall(r'chr=(\d+)', description)[0])
        else:
            strands, starts, ends, chromosomes = [], [], [], []
            for id_r, id_o in s_id:
                description = train_data[id_r]
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

    aux['node_attrs'].extend(['read_strand', 'read_start', 'read_end', 'read_chr'])

    read_strand_list, read_start_list, read_end_list, read_chr_list = [read_strands[i] for i in range(n_nodes)], [read_starts[i] for i in range(n_nodes)], [read_ends[i] for i in range(n_nodes)], [read_chrs[i] for i in range(n_nodes)]
    g['read_strand'] = torch.tensor(read_strand_list)
    g['read_start'] = torch.tensor(read_start_list)
    g['read_end'] = torch.tensor(read_end_list)
    g['read_chr'] = torch.tensor(read_chr_list)
    aux['read_strand_dict'] = read_strands
    aux['read_start_dict'] = read_starts
    aux['read_end_dict'] = read_ends
    aux['read_chr_dict'] = read_chrs

    nx_g = to_networkx(data=g, node_attrs=aux['node_attrs'], edge_attrs=aux['edge_attrs'])

    unique_chrs = set(read_chrs.values())
    if len(unique_chrs) == 1:
        ms_pos, labels = process_graph(nx_g)
    else:
        ms_pos, labels = process_graph_combo(nx_g)
    nx.set_edge_attributes(nx_g, labels, 'y')
    labels = torch.tensor([data['y'] for _, _, data in nx_g.edges(data=True)])
    g['y'] = labels
    aux['edge_attrs'].append('y')

    return g, aux
