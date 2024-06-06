from Bio import SeqIO
from Bio.Seq import Seq
import gzip
from tqdm import tqdm

def parse_fasta(path):
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
        for read in tqdm(SeqIO.parse(handle, filetype), ncols=120):
            id = read.description.split()[0]
            data[id] = (read.seq, str(Seq(read.seq).reverse_complement()))

    return data
            
