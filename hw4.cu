#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#include <cuda.h>

#define SRAMSIZE 49152  // 49152, br = 384 or 192
#define NUMPAD 128
#define BR 32
#define BC 32

void input(char *input_filename);
void output(char *output_filename);
int ceil(int a, int b);

__global__ void dev_flash_attention(float* Q, float* K, float* V, float* O, float* l, float* m, int B, int N, int d, int bc, int br, int tc, int tr, float scaler);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d, embedding_dimension;
float *Q, *K, *V, *O, scaler;


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    double start, end;
    start = getTimeStamp();

    input(argv[1]);



    // Initialize Q, K, V in HBM
    float *dev_Q, *dev_K, *dev_V;
    cudaMalloc(&dev_Q, B * N * d * sizeof(float));
    cudaMemcpy(dev_Q, Q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_K, B * N * d * sizeof(float));
    cudaMemcpy(dev_K, K, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_V, B * N * d * sizeof(float));
    cudaMemcpy(dev_V, V, B * N * d * sizeof(float), cudaMemcpyHostToDevice);

    // accumulation buffers (used for each batch)
    float *l = (float *)malloc(N * sizeof(float));  // Accumulates scaling factors
    float *m = (float *)malloc(N * sizeof(float));  // Tracks maximum values for numerical stability
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }
    // Step 2: Initialize O, l, m in HBM
    float *dev_O, *dev_l, *dev_m;
    cudaMalloc(&dev_O, B * N * d * sizeof(float));
    cudaMemcpy(dev_O, O, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_l, B * N * sizeof(float));
    cudaMemcpy(dev_l, l, B * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_m, B * N * sizeof(float));
    cudaMemcpy(dev_m, m, B * N * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Define block sizes for tiling
    // int br = SRAMSIZE / (4 * d), bc = d;  // br = 384 or 192, bc = 32 or 64, br * bc = M / 4 = 12288, ensure sufficient shared memory for 4 arrays, but no sufficient threads
    // max #threads = 1024, (1024 / 32, 32) = (32, 32), (1024 / 64, 64) = (16, 64)
    // int tr = ceil(N, br), tc = ceil(N, bc)
    // tr, tc: max_N = 32768, 32768 / 384 = 85.3 = 86, tc = 32768 / 32 = 1024
    int br = BR, bc = BC;
    int tr = N / br, tc = N / bc;  // tr = 4, 8, 16, 32, 64, 128, 256, 512, 1024 (32768 / 32), how many column for each set
    // dim3 threads(br, bc);          // (32, 32)
    dim3 threads(tc);
    dim3 blocks(B);
    // dim3 blocks(B, tr);   // (batch size, tr) -> only need tc times for each thread
    scaler = 1.0 / sqrt(d);

    // Calculate requested SRAM
    const int sram_size = (2 * bc * d * sizeof(float) + 1 * br * d * sizeof(float) + 1 * br * bc * sizeof(float));
    printf("requested shared memory: %d\n", sram_size);

    dev_flash_attention<<<blocks, threads, SRAMSIZE>>>(dev_Q, dev_K, dev_V, dev_O, dev_l, dev_m, B, N, d, bc, br, tc, tr, scaler);

    // for (int i = 0; i < B; i++) {
    //     flash_attention(
    //         Q + (i * N * d),
    //         K + (i * N * d),
    //         V + (i * N * d),
    //         O + (i * N * d)
    //     );
    // }

    end = getTimeStamp();
    // printf("(B, N, d): (%d, %d, %d)\n", B, N, embedding_dimension);
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    // fread(&embedding_dimension, sizeof(int), 1, file);
    // d = (embedding_dimension % NUMPAD) ? embedding_dimension : (embedding_dimension / NUMPAD + 1) * NUMPAD;
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void dev_flash_attention(float* Q, float* K, float* V, float* O, float* l, float* m, int B, int N, int d, int bc, int br, int tc, int tr, float scaler) {
    if (blockIdx.x * N * d + threadIdx.x * bc >= N) {
        return;
    }

    int tc_start = threadIdx.x;
    // int bc_start = threadIdx.y;
    // int batch_start = blockIdx.x * blockDim.x;
    int batch_start = blockIdx.x;
    int kv_offset;
    int qo_offset;
    int lm_offset;

    // Step 3: Declare device variables for blocks of kj, vj, qi, and oi
    extern __shared__ float shared[];
    float *kj = shared;       // kj[d][bc]
    float* vj = kj + bc * d;  // vj[bc][d]
    float* qi = vj + bc * d;  // qi[br][d]
    float* s = qi + bc * d;   // s[br][bc]

    // Step 4: Declare device variables for blocks of li, mi, sij, pij, mij, lij, mi_new, li_new
    float *li = s + br * bc;  // li[br]
    float *mi = li + br;      // mi[br]
    // float sij;
    float pv;
    // float pij;
    float *mij = mi + br;   // mij[br]
    float *lij = mij + br;  // lij[br]
    float mi_new;
    float li_new;
    float sum;
    // for (int r = 0; r < br; ++r) {
    //     mi[r] = FLT_MIN;
    //     li[r] = FLT_MIN;
    // }

    // Outer loop: Iterate over blocks of K and V
    // for (int j = 0; j < tc; j++) {
        // Load block of K and V from HBM to SRAM
        // memcpy(kj, k + j * bc * d, bc * d * sizeof(float));
        // memcpy(vj, v + j * bc * d, bc * d * sizeof(float));
        kv_offset = (batch_start * N * d) + (tc_start * bc * d);

        for (int c = 0; c < bc; ++c) {
            for (int idx = 0; idx < d; ++idx) {
                kj[c * d + idx] = K[kv_offset + c * d + idx];
                vj[c * d + idx] = V[kv_offset + c * d + idx];
            }
        }
        __syncthreads();

        // Inner loop: Iterate over blocks of Q
        for (int i = 0; i < tr; i++) {
            // Load block of Q, O to SRAM and mi, li to registers
            // memcpy(qi, q + i * br * d, br * d * sizeof(float));
            // memcpy(oi, o + i * br * d, br * d * sizeof(float));
            // memcpy(li, l + i * br, br * sizeof(float));
            // memcpy(mi, m + i * br, br * sizeof(float));
            qo_offset = (batch_start * N * d) + (i * br * d);
            lm_offset = (batch_start * N) + (i * br);
            for (int r = 0; r < br; ++r) {
                for (int idx = 0; idx < d; ++idx) {
                    qi[r * d + idx] = Q[qo_offset + r * d + idx];
                    // oi[r * d + idx] = O[qo_offset + r * d + idx];
                }
                mi[r] = m[lm_offset + r];
                li[r] = l[lm_offset + r];
            }
            __syncthreads();
            // lm_offset = batch_start * N + i * br + bc_start;
            // mi = m[lm_offset];
            // li = l[lm_offset];

            // Compute scaled dot-product of Q and K blocks
            // QKDotAndScalar(sij, qi, kj, br, bc, 1.0 / sqrt(d));  // Kernel fusion
            // Softmax computation over the block
            // RowMax(mij, sij, br, bc);  // Find row-wise maximum for numerical stability
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    // mij[r] = FLT_MIN;
                    sum = 0;
                    for (int idx = 0; idx < d; ++idx) {
                        sum += qi[r * d + idx] * kj[c * d + idx];
                    }
                    sum *= scaler;
                    s[(r * bc) + c] = sum;
                    // mij[r] = max(mij[r], sum);
                }
            }

            for (int r = 0; r < br; ++r) {
                mij[r] = s[r * bc];
                for (int c = 0; c < bc; ++c) {
                    mij[r] = max(mij[r], s[r * bc + c]);
                }
            }

            // MinusMaxAndExp(pij, sij, mij, br, bc);  // Subtract max and exponentiate
            // RowSum(lij, pij, br, bc);  // Compute row-wise sum for normalization
            for (int r = 0; r < br; ++r) {
                // lij[r] = 0;
                for (int c = 0; c < bc; ++c) {
                    s[(r * bc) + c] = expf(s[(r * bc) + c] - mij[r]);
                    // lij[r] += s[(r * bc) + c];
                }
            }
            for (int r = 0; r < br; ++r) {
                lij[r] = 0;
                for (int c = 0; c < bc; ++c) {
                    lij[r] += s[(r * bc) + c];
                }
            }

            // Update running max, sum, and output blocks
            // UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, br, bc);
            for (int r = 0; r < br; ++r) {
                mi_new = max(mi[r], mij[r]);
                li_new = (expf(mi[r] - mi_new) * li[r]) + (expf(mij[r] - mi_new) * lij[r]);
                for (int idx = 0; idx < d; ++idx) {
                    pv = 0.0F;
                    for (int c = 0; c < bc; ++c) {
                        pv += s[(r * bc) + c] * vj[c * d + idx];
                    }
                    // oi[i * d + j] = (li[i] * exp(mi[i] - mi_new[i]) * oi[i * d + j] + exp(mij[i] - mi_new[i]) * pv) / li_new[i];
                    O[qo_offset + r * d + idx] = ((li[r] * expf(mi[r] - mi_new) * O[qo_offset + r * d + idx]) + \
                                                  (expf(mij[r] - mi_new) * pv)) / li_new;
                }
                m[lm_offset + r] = mi_new;
                l[lm_offset + r] = li_new;
                // mi[r] = mi_new;
                // li[r] = li_new;
            }

            // Store updated output and normalization factors back to global memory
            // memcpy(o + i * br * d, oi, br * d * sizeof(float));
            // memcpy(l + i * br, li, br * sizeof(float));
            // memcpy(m + i * br, mi, br * sizeof(float));
        }
        __syncthreads();
    // }
}
