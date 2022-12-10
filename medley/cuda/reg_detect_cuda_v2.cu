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

/*
static void kernel_reg_detect(int niter, int maxgrid, int length,int* sum_tang,int* mean,
                              int* path, int* diff, int* sum_diff)
{
  clock_t begin = clock();

  for (int t = 0; t < niter; t++)
  { 
    for (j = 0; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        for (cnt = 0; cnt <= length - 1; cnt++)
          diff[(j*maxgrid + i)*length + cnt] = sum_tang[j*maxgrid + i];  

    for (j = 0; j <= maxgrid - 1; j++)
    {
      for (i = j; i <= maxgrid - 1; i++)
      {
        sum_diff[(j*maxgrid + i)*length] = diff[(j*maxgrid + i)*length];

        for (cnt = 1; cnt <= length - 1; cnt++)
          sum_diff[(j*maxgrid + i)*length + cnt] = sum_diff[(j*maxgrid + i)*length + cnt - 1] + diff[(j*maxgrid + i)*length + cnt];
        mean[j*maxgrid + i] = sum_diff[(j*maxgrid + i)*length + (length - 1)];
      }
    }

    for (i = 0; i <= maxgrid - 1; i++)
      path[i] = mean[i];

    for (j = 1; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        path[j*maxgrid + i] = path[(j - 1)*maxgrid + (i-1)] + mean[j*maxgrid + i];
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}
*/

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

static void kernel_reg_detect_cuda(int niter, int maxgrid, int length,int* sum_tang_d,int* mean_d,
                              int* path_d, int* diff_d, int* sum_diff_d)
{
  clock_t begin = clock();

  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid((maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE,(maxgrid+BLOCK_SIZE-1)/BLOCK_SIZE);

  //dim3 dimBlock(maxgrid,maxgrid);
  //dim3 dimGrid(1, 1);

  // Allocazioni in memoria
  for (int t = 0; t < niter; t++)
  { 
    
    int a = 0;

    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 1);

    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 2);

    kernel<<<dimGrid, dimBlock>>>(maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d, 3);

    cudaDeviceSynchronize();
    for (int j = 1; j <= maxgrid - 1; j++)
      for (int i = j; i <= maxgrid - 1; i++)
        path_d[j*maxgrid + i] = path_d[(j - 1)*maxgrid + (i-1)] + mean_d[j*maxgrid + i];
   
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

/*

  Modalità UVM

*/
int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    int* sum_tang = (int*)malloc(sizeof(int) * maxgrid * maxgrid);


    //-------------------------------------------------------------------

    //int* diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);

    //int* sum_diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    //-------------------------------------------------------------------


    int* sum_tang_d;
    cudaMalloc((void**)&sum_tang_d, sizeof(int) * maxgrid * maxgrid);

    int* mean_d_uva;
    cudaMallocManaged((void**)&mean_d_uva, sizeof(int) * maxgrid * maxgrid);
    
    int* path_d_uva;
    cudaMallocManaged(&path_d_uva, sizeof(int) * maxgrid * maxgrid);

    int* diff_d;
    cudaMalloc((void**)&diff_d, sizeof(int) * maxgrid * maxgrid * length);

    int* sum_diff_d;
    cudaMalloc((void**)&sum_diff_d, sizeof(int) * maxgrid * maxgrid * length);
    
    init_array(maxgrid, sum_tang, mean_d_uva, path_d_uva);

    // Copia su GPU
    cudaMemcpy(sum_tang_d, sum_tang, sizeof(int) * maxgrid * maxgrid, cudaMemcpyHostToDevice);

    //print_array(maxgrid, path_d_uva);
    
    //kernel_reg_detect(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff);

    //kernel_reg_detect(niter, maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d);

    kernel_reg_detect_cuda(niter, maxgrid, length, sum_tang_d, mean_d_uva, path_d_uva, diff_d, sum_diff_d);
    cudaDeviceSynchronize();


    //print_array(maxgrid, path_d_uva);



    free(sum_tang);

    cudaFree(sum_tang_d);
    cudaFree(mean_d_uva);
    cudaFree(path_d_uva);
    cudaFree(diff_d);
    cudaFree(sum_diff_d);
    
    
}
