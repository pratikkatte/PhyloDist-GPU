#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>

__global__ void pairwise_hamming(
    const unsigned char* __restrict__ seqs,
    int N, int M,
    float* __restrict__ out
);

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./hamming <binfile> <shapefile>\n";
        return 1;
    }

    int N, M;
    {
        FILE* s = fopen(argv[2], "r");
        if (!s) { perror("shape file"); return 1; }
        fscanf(s, "%d %d", &N, &M);
        fclose(s);
    }

    size_t S = (size_t)N * M;
    unsigned char* hseqs = (unsigned char*)malloc(S);
    {
        FILE* f = fopen(argv[1], "rb");
        if (!f) { perror("bin file"); return 1; }
        fread(hseqs, 1, S, f);
        fclose(f);
    }

    unsigned char *dseqs;
    float *dout;
    cudaMalloc(&dseqs, S);
    cudaMalloc(&dout, (size_t)N * N * sizeof(float));

    cudaMemcpy(dseqs, hseqs, S, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, (size_t)N * N * sizeof(float));


    dim3 grid(N, N);
    int tpb = 256;
    size_t shmem = 2 * tpb * sizeof(int);

    pairwise_hamming<<<grid, tpb, shmem>>>(dseqs, N, M, dout);
    cudaDeviceSynchronize();


    float* D = (float*)malloc((size_t)N * N * sizeof(float));
    cudaMemcpy(D, dout, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);

    
    FILE* out = fopen("gpu_dist.bin", "wb");
    fwrite(D, sizeof(float), (size_t)N * N, out);
    fclose(out);

    
    free(hseqs);
    free(D);
    cudaFree(dseqs);
    cudaFree(dout);

    std::cerr << "GPU computation done\n";
    return 0;
}

__global__ void pairwise_hamming(
    const unsigned char* __restrict__ seqs,
    int N, int M,
    float* __restrict__ out
) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= N || j >= N || j <= i) return;

    int mism = 0, valid = 0;

    const unsigned char* a = seqs + i * M;
    const unsigned char* b = seqs + j * M;

    for (int k = threadIdx.x; k < M; k += blockDim.x) {
        unsigned char x = a[k], y = b[k];
        if (x == 255 || y == 255) continue;  // skip gaps
        valid++;
        mism += (x != y);
    }

    // reduction in shared memory
    extern __shared__ int smem[];
    int* mismem = smem;
    int* valmem = smem + blockDim.x;

    mismem[threadIdx.x] = mism;
    valmem[threadIdx.x] = valid;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mismem[threadIdx.x] += mismem[threadIdx.x + s];
            valmem[threadIdx.x] += valmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float pd = valmem[0] ? float(mismem[0]) / float(valmem[0]) : 0.0f;
        out[(size_t)i * N + j] = pd;
        out[(size_t)j * N + i] = pd;
    }
}
