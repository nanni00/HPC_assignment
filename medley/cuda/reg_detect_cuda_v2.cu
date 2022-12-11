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


__global__ void kernel(int maxgrid, int length, 
                      int* sum_tang_uva,int* mean_uva, int* path_uva,
                      int* diff_uva, int* sum_diff_uva, int selector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= maxgrid || j >= maxgrid)
    return;
    
  if (selector == 1) {
    //Primo for
    for (int cnt = 0; cnt <= length - 1; cnt++)
      diff_uva[(j*maxgrid + i)*length + cnt] = sum_tang_uva[j*maxgrid + i];  

  } else if (selector == 2) {
    //Secondo for
    sum_diff_uva[(j*maxgrid + i)*length] = diff_uva[(j*maxgrid + i)*length];

    for (int cnt = 1; cnt <= length - 1; cnt++)
      sum_diff_uva[(j*maxgrid + i)*length + cnt] = sum_diff_uva[(j*maxgrid + i)*length + cnt - 1] + diff_uva[(j*maxgrid + i)*length + cnt];
    mean_uva[j*maxgrid + i] = sum_diff_uva[(j*maxgrid + i)*length + (length - 1)];

  } else if (selector == 3 && j == 0) {
    //Terzo for
    path_uva[i] = mean_uva[i];

  } else if (selector == 4 && i == 0 && j == 0) {
    //Quarto for
    for (j = 1; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        path_uva[j*maxgrid + i] = path_uva[(j - 1)*maxgrid + (i-1)] + mean_uva[j*maxgrid + i];
  }
}

#define BLOCK_SIZE 32

static void kernel_reg_detect_cuda(int niter, int maxgrid, int length,int* sum_tang_d,int* mean_d,
                              int* path_d, int* diff_d, int* sum_diff_d)
{
  clock_t begin = clock();

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid((maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE,(maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE);

  for (int t = 0; t < niter; t++)
  {   
    // first 3-nested-for
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 1);

    // second 3-nested-for
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 2);
    
    // 1-for
    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 3);

    // sequential part
    cudaDeviceSynchronize();
    for (int j = 1; j <= maxgrid - 1; j++)
      for (int i = j; i <= maxgrid - 1; i++)
        path_d[j*maxgrid + i] = path_d[(j - 1)*maxgrid + (i-1)] + mean_d[j*maxgrid + i];
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

/**
 * Version 3.2
 * Usage of Unified Virtual Memory (UVM)
 * The kernel is a monolythic function with a "multiplexer",
 * in order to choose the correct part of the algorithm.
*/
int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    int* sum_tang_uva;
    int* sum_diff_uva;
    int* path_uva;
    int* mean_uva;
    int* diff_uva;
    
    cudaMallocManaged(&sum_diff_uva, sizeof(int) * maxgrid * maxgrid * length);
    cudaMallocManaged(&sum_tang_uva, sizeof(int) * maxgrid * maxgrid);
    cudaMallocManaged(&mean_uva, sizeof(int) * maxgrid * maxgrid);
    cudaMallocManaged(&path_uva, sizeof(int) * maxgrid * maxgrid);
    cudaMallocManaged(&diff_uva, sizeof(int) * maxgrid * maxgrid * length);
    
    init_array(maxgrid, sum_tang_uva, mean_uva, path_uva);

    print_array(maxgrid, path_uva);
    
    kernel_reg_detect_cuda(niter, maxgrid, length, sum_tang_uva, mean_uva, path_uva, diff_uva, sum_diff_uva);
    cudaDeviceSynchronize();

    print_array(maxgrid, path_uva);

    cudaFree(sum_tang_uva);
    cudaFree(sum_diff_uva);
    cudaFree(mean_uva);
    cudaFree(path_uva);
    cudaFree(diff_uva);
    
    return 0;
}
