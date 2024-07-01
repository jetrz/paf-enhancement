import subprocess

# for chr in [1, 3, 5, 9, 11, 12, 16, 17, 18, 19, 20]:
#     for n in range(10, 15):

for chr in [11, 16, 17, 19, 20]:
    for n in range(0, 5):
        command = f"../GitHub/hifiasm/hifiasm --prt-raw --write-paf -o chr{chr}_M_{n}_asm -t16 -l0 /mnt/sod2-project/csb4/wgs/lovro_interns/joshua/centrom_dataset_2/full_reads/chr{chr}_MATERNAL_centromere_{n}.fasta"
        result = subprocess.run(command, shell=True)

        print("chr", chr, "n", n, "done!")

        
    command = f"mv chr{chr}_* ../../../mnt/sod2-project/csb4/wgs/lovro_interns/joshua/paf-enhancement/datasets/"
    result = subprocess.run(command, shell=True)
