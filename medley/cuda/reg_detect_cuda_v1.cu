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

__global__ void kernel(int maxgrid, int length,int* sum_tang_d,int* mean_d, int* path_d, int* diff_d, int* sum_diff_d, int selector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= maxgrid || j >= maxgrid)
    return;
    
  if (selector == 1) {
    //Primo for
    for (int cnt = 0; cnt <= length - 1; cnt++)
      diff_d[(j*maxgrid + i)*length + cnt] = sum_tang_d[j*maxgrid + i];  
      
  } else if (selector == 2) {
    //Secondo for
    sum_diff_d[(j*maxgrid + i)*length] = diff_d[(j*maxgrid + i)*length];

    for (int cnt = 1; cnt <= length - 1; cnt++)
      sum_diff_d[(j*maxgrid + i)*length + cnt] = sum_diff_d[(j*maxgrid + i)*length + cnt - 1] + diff_d[(j*maxgrid + i)*length + cnt];
    mean_d[j*maxgrid + i] = sum_diff_d[(j*maxgrid + i)*length + (length - 1)];
  
  } else if (selector == 3 && j == 0) {
    //Terzo for
    path_d[i] = mean_d[i];

  } else if (selector == 4 && i == 0 && j == 0) {
    //Quarto for
    for (j = 1; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        path_d[j*maxgrid + i] = path_d[(j - 1)*maxgrid + (i-1)] + mean_d[j*maxgrid + i];
  }
}

#define BLOCK_SIZE 32

static void kernel_reg_detect(int niter, int maxgrid, int length,int* sum_tang_d,int* mean_d,
                              int* path_d, int* diff_d, int* sum_diff_d)
{

  //dim3 dimGrid((maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE,(maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE);
  //dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

  dim3 dimGrid(1, 1);
  dim3 dimBlock(maxgrid,maxgrid);

  clock_t begin = clock();
  
  for (int t = 0; t < niter; t++)
  { 
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 1);
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 2);
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 3);
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 4);
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

/**
 * Tutto viene eleborato sulla GPU e poi copiato su host.
*/
int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    int* sum_tang = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    //int* sum_diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    //int* diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    int* mean = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* path = (int*)malloc(sizeof(int) * maxgrid * maxgrid);

    int* sum_tang_d;
    int* sum_diff_d;
    int* mean_d;
    int* path_d;
    int* diff_d;
    
    // allocation on device
    cudaMalloc((void**)&sum_tang_d, sizeof(int) * maxgrid * maxgrid);
    cudaMalloc((void**)&sum_diff_d, sizeof(int) * maxgrid * maxgrid * length);
    cudaMalloc((void**)&mean_d, sizeof(int) * maxgrid * maxgrid);
    cudaMalloc((void**)&path_d, sizeof(int) * maxgrid * maxgrid);
    cudaMalloc((void**)&diff_d, sizeof(int) * maxgrid * maxgrid * length);

    init_array(maxgrid, sum_tang, mean, path);
    
    print_array(maxgrid, path);

    // copies to the device
    cudaMemcpy(sum_tang_d, sum_tang, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);
    cudaMemcpy(mean_d, mean, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);
    cudaMemcpy(path_d, path, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);

    // call the function
    kernel_reg_detect(niter, maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d);

    // copy to the host
    cudaMemcpy(path, path_d, sizeof(int) * maxgrid * maxgrid, cudaMemcpyDeviceToHost);

    print_array(maxgrid, path);

    // free host memory
    free(sum_tang);
    free(mean);
    free(path);

    cudaFree(sum_tang_d);
    cudaFree(mean_d);
    cudaFree(path_d);
    cudaFree(diff_d);
    cudaFree(sum_diff_d);

    return 0;
}
