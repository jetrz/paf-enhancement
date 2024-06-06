import gzip
from Bio import SeqIO

if __name__ == "__main__":
    filetype = 'fastq'
    with gzip.open('CRR302668_p0.22.fastq.gz', 'rt') as handle:
        reads = SeqIO.parse(handle, filetype)

        read_headers = {read.id: read.description for read in reads}
        print(len(read_headers))
