# scripts/encode_fasta.py
import sys, numpy as np

map4 = {ord('A'):0, ord('C'):1, ord('G'):2, ord('T'):3, ord('-'):255}

name, seqs = None, []
file_name = sys.argv[1]

for line in open(file_name):
    line = line.strip()
    if not line: continue
    if line.startswith('>'):
        name = line[1:]
    else:
        if name:
            seqs.append(line)
            name = None

N, M = len(seqs), len(seqs[0])
arr = np.empty((N,M), dtype=np.uint8)

for i,s in enumerate(seqs):
    assert len(s)==M
    arr[i,:] = np.frombuffer(s.encode(), dtype=np.uint8)

arr = np.vectorize(lambda b: map4.get(b,255))(arr)
arr.tofile(sys.argv[2])     # raw binary row-major (N*M bytes)
open(sys.argv[2]+".shape","w").write(f"{N} {M}\n")
print("wrote", sys.argv[2], "shape", N, M)