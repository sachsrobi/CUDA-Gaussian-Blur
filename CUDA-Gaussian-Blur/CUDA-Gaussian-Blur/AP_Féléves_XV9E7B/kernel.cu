#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 16

// Gauss Kernel
__constant__ float d_kernel[9]; // GPU-n konstans memória

float h_kernel[9] =
{
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
    2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
    1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f
};

// CPU IMPLEMENTÁCIÓ
void CPU_GaussBlur(unsigned char* outImg, const unsigned char* inImg, int W, int H)
{
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int yy = y + ky;
                    int xx = x + kx;

                    // határkezelés
                    if (yy < 0 || yy >= H) yy = y;
                    if (xx < 0 || xx >= W) xx = x;

                    sum += inImg[yy * W + xx] * h_kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }

            outImg[y * W + x] = (unsigned char)sum;
        }
    }
}

// GPU SHARED MEMORY OPTIMALIZÁCIÓ
__global__ void GaussBlur_Shared(unsigned char* outImg, const unsigned char* inImg, int W, int H)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // --- clampelt indexek a középső pixelhez ---
    int cx = min(max(x, 0), W - 1);
    int cy = min(max(y, 0), H - 1);

    tile[ty][tx] = inImg[cy * W + cx];

    // bal halo
    if (threadIdx.x == 0)
    {
        int lx = min(max(x - 1, 0), W - 1);
        tile[ty][0] = inImg[cy * W + lx];
    }

    // jobb halo
    if (threadIdx.x == BLOCK_SIZE - 1)
    {
        int rx = min(max(x + 1, 0), W - 1);
        tile[ty][BLOCK_SIZE + 1] = inImg[cy * W + rx];
    }

    // felső halo
    if (threadIdx.y == 0)
    {
        int ty_cl = min(max(y - 1, 0), H - 1);
        tile[0][tx] = inImg[ty_cl * W + cx];
    }

    // alsó halo
    if (threadIdx.y == BLOCK_SIZE - 1)
    {
        int by_cl = min(max(y + 1, 0), H - 1);
        tile[BLOCK_SIZE + 1][tx] = inImg[by_cl * W + cx];
    }

    // sarkok (mind a 4)
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        int lx = min(max(x - 1, 0), W - 1);
        int ty_cl = min(max(y - 1, 0), H - 1);
        tile[0][0] = inImg[ty_cl * W + lx];
    }

    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0)
    {
        int rx = min(max(x + 1, 0), W - 1);
        int ty_cl = min(max(y - 1, 0), H - 1);
        tile[0][BLOCK_SIZE + 1] = inImg[ty_cl * W + rx];
    }

    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1)
    {
        int lx = min(max(x - 1, 0), W - 1);
        int by_cl = min(max(y + 1, 0), H - 1);
        tile[BLOCK_SIZE + 1][0] = inImg[by_cl * W + lx];
    }

    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1)
    {
        int rx = min(max(x + 1, 0), W - 1);
        int by_cl = min(max(y + 1, 0), H - 1);
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = inImg[by_cl * W + rx];
    }

    __syncthreads();

    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            sum += tile[ty + ky][tx + kx] * d_kernel[(ky + 1) * 3 + (kx + 1)];

    outImg[y * W + x] = (unsigned char)sum;
}

int main()
{
    const int W = 4096;
    const int H = 4096;
    const int imgSize = W * H;

    unsigned char* h_input = (unsigned char*)malloc(imgSize);
    unsigned char* h_cpuOut = (unsigned char*)malloc(imgSize);
    unsigned char* h_gpuOut = (unsigned char*)malloc(imgSize);

    // random input kép
    for (int i = 0; i < imgSize; i++)
        h_input[i] = rand() % 256;

    // Gauss kernel feltöltése GPU-ra
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * 9);

    unsigned char* d_in, * d_out;
    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);

    cudaMemcpy(d_in, h_input, imgSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CPU idő mérés
    auto t1 = std::chrono::high_resolution_clock::now();
    CPU_GaussBlur(h_cpuOut, h_input, W, H);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("CPU time: %.3f ms\n", std::chrono::duration<double, std::milli>(t2 - t1).count());

    // GPU SHARED
    t1 = std::chrono::high_resolution_clock::now();
    GaussBlur_Shared << <grid, block >> > (d_out, d_in, W, H);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_gpuOut, d_out, imgSize, cudaMemcpyDeviceToHost);
    printf("GPU shared time: %.3f ms\n", std::chrono::duration<double, std::milli>(t2 - t1).count());

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_input);
    free(h_cpuOut);
    free(h_gpuOut);

    cudaDeviceReset();
    return 0;
}
