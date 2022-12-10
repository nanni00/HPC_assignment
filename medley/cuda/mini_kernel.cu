#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* Include benchmark-specific header. */
/* Default data type is int, default size is 50. */
#include "reg_detect.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

static void init_array(int maxgrid, int* sum_tang, int* mean, int* path)
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++)
    {
      sum_tang[i * maxgrid + j] = (int)((i + 1) * (j + 1));
      mean[i * maxgrid + j] = ((int)i - j) / maxgrid;
      path[i * maxgrid + j] = ((int)i * (j - 1)) / maxgrid;
    }
}

static void print_array(int maxgrid, int* path)
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++)
    {
      fprintf(stdout, "%d ", path[i * maxgrid + j]);
      if ((i * maxgrid + j) % 20 == 0)
        fprintf(stdout, "\n");
    }
  fprintf(stdout, "\n");
}

__device__ void kernel_1(int maxgrid, int length, int* diff, int* sum_tang)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= maxgrid-1 && j <= maxgrid-1)
    {
        for (int cnt = 0; cnt <= length - 1; cnt++)
          diff[(j*maxgrid + i)*length + cnt] = sum_tang[j*maxgrid + i];
    }
}

__device__ void kernel_2(int maxgrid, int length, int* sum_diff, int*diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i <= maxgrid-1 && j <= maxgrid-1)
    {
        sum_diff[(j*maxgrid + i)*length] = diff[(j*maxgrid + i)*length];
    }
}

__device__ void kernel_3(int maxgrid, int length, int* sum_diff, int* diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i <= maxgrid-1 && j <= maxgrid-1)
    {
        for (int cnt = 1; cnt <= length - 1; cnt++)
        {
            sum_diff[(j*maxgrid + i)*length + cnt] = sum_diff[(j*maxgrid + i)*length + cnt - 1] + diff[(j*maxgrid + i)*length + cnt];
        }
    }
}

__device__ void kernel_4(int maxgrid, int length, int* mean, int* sum_diff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i <= maxgrid-1 && j <= maxgrid-1)
    {
        mean[j*maxgrid + i] = sum_diff[(j*maxgrid + i)*length + (length - 1)];
    }
}

__device__ void kernel_5(int maxgrid, int* path, int* mean)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<=maxgrid-1)
    {
        path[i] = mean[i];
    }
}

__device__ void kernel_6(int maxgrid, int* path, int* mean)
{
    for (int j = 1; j <= maxgrid - 1; j++){
      for (int i = j; i <= maxgrid - 1; i++)
      {
        path[j*maxgrid + i] = path[(j - 1)*maxgrid + (i-1)] + mean[j*maxgrid + i];
      }
    }
}

__global__ void main_kernel(int niter, int maxgrid, int length,int* sum_tang_d,int* mean_d,
                              int* path_d, int* diff_d, int* sum_diff_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
        if(i<niter)
        {
            kernel_1(maxgrid, length, diff_d, sum_tang_d);
            __syncthreads();
            kernel_2(maxgrid, length, sum_diff_d, diff_d);
            __syncthreads();
            kernel_3(maxgrid, length, sum_diff_d, diff_d);
            __syncthreads();
            kernel_4(maxgrid, length, mean_d, sum_diff_d);
            __syncthreads();
            kernel_5(maxgrid, path_d, mean_d);
            __syncthreads();
            kernel_6(maxgrid, path_d, mean_d);
            __syncthreads();
        }
        //printf("i: %d\n", i);
}

void reg_detect(int niter, int maxgrid, int length,int* sum_tang,int*mean,
                              int* path, int* diff, int* sum_diff)
{
    //Allocazione array su device
    int* sum_tang_d;
    cudaMalloc((void**)&sum_tang_d, sizeof(int) * maxgrid * maxgrid);

    int* mean_d;
    cudaMalloc((void**)&mean_d, sizeof(int) * maxgrid * maxgrid);
    
    int* path_d;
    cudaMalloc((void**)&path_d, sizeof(int) * maxgrid * maxgrid);

    int* diff_d;
    cudaMalloc((void**)&diff_d, sizeof(int) * maxgrid * maxgrid * length);

    int* sum_diff_d;
    cudaMalloc((void**)&sum_diff_d, sizeof(int) * maxgrid * maxgrid * length);

    //copia dati su GPU
    cudaMemcpy(sum_tang_d, sum_tang, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);
    cudaMemcpy(mean_d, mean, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);
    cudaMemcpy(path_d, path, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);
    cudaMemcpy(diff_d, diff, sizeof(int) * maxgrid * maxgrid * length, cudaMemcpyHostToDevice);
    cudaMemcpy(sum_diff_d, sum_diff, sizeof(int) * maxgrid * maxgrid * length, cudaMemcpyHostToDevice);

    dim3 BlocksDim (1,1);
    dim3 ThreadsPerBlock ( maxgrid, maxgrid);
    //dim3 ThreadsPerBlock(max, 32);
	//dim3 BlocksDim((unsigned int)ceil( ((float)maxgrid) / ((float)block.x) ), (unsigned int)ceil( ((float)maxgrid) / ((float)block.y) ));

    clock_t begin = clock();

    //________________________________Scrivere chiamate ai kernel_____________________________
    main_kernel<<<BlocksDim,ThreadsPerBlock>>>(niter, maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d);
    
    clock_t end = clock();
    printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    cudaMemcpy(path, path_d, sizeof(int) * maxgrid * maxgrid, cudaMemcpyDeviceToHost);

    cudaFree(sum_tang_d);
    cudaFree(mean_d);
    cudaFree(path_d);
    cudaFree(diff_d);
    cudaFree(sum_diff_d);


}


int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    //Allocazione array su host
    int* sum_tang = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* mean = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* path = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    int* sum_diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    
    init_array(maxgrid, sum_tang, mean, path);
    print_array(maxgrid, path);

    reg_detect(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff);

    print_array(maxgrid, path);



    free(sum_tang);
    free(mean);
    free(path);


    
    
}