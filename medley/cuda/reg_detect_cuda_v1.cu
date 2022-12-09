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

static void kernel_reg_detect(int niter, int maxgrid, int length,int* sum_tang,int* mean,
                              int* path, int* diff, int* sum_diff)
{
  clock_t begin = clock();
  int t, i, j, cnt;

  for (t = 0; t < niter; t++)
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

/*
static void kernel_reg_detect(int niter, int maxgrid, int length,int** sum_tang_d,int**mean_d,
                              int** path_d, int*** diff_d, int*** sum_diff_d)
{
  clock_t begin = clock();
  int t, i, j, cnt;

  // Allocazioni in memoria
  for (t = 0; t < niter; t++)
  { 
    
    int a = 0;

    kernel_1<<<...>>>(..., 1);

    kernel_1<<<...>>>(..., 2);

    kernel_1<<<...>>>(..., 3);

    kernel_1<<<...>>>(..., 4);
    
    ;
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}
*/

__global__ void kernel_1(int maxgrid, int length,int** sum_tang,int**mean, int*** diff, int*** sum_diff, int** path, int selector) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= maxgrid || j >= maxgrid)
    return;
  
  if (selector == 1) {
    //Primo for
    for (int cnt = 0; cnt <= length - 1; cnt++)
      diff[j][i][cnt] = sum_tang[j][i]; 
  } else if (selector == 2) {
    //Secondo for
    sum_diff[j][i][0] = diff[j][i][0];

    for (int cnt = 1; cnt <= length - 1; cnt++)
      sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
    mean[j][i] = sum_diff[j][i][length - 1];
  } else if (selector == 3) {
    //Terzo for
    path[0][i] = mean[0][i];
  } else if (selector == 4 && i == 0 && j == 0) {
    //Quarto for
    for (j = 1; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        path[j][i] = path[j - 1][i - 1] + mean[j][i];
  }
}


int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    int* sum_tang = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* mean = (int*)malloc(sizeof(int) * maxgrid * maxgrid);
    int* path = (int*)malloc(sizeof(int) * maxgrid * maxgrid);


    //-------------------------------------------------------------------

    int* diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);

    int* sum_diff = (int*)malloc(sizeof(int) * maxgrid * maxgrid * length);
    //-------------------------------------------------------------------


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
    
    init_array(maxgrid, sum_tang, mean, path);
    print_array(maxgrid, path);
    
    kernel_reg_detect(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff);

    //gpuErrchk(cudaMemcpy2D(sum_tang_d, sizeof(int) * maxgrid, sum_tang, sizeof(int) * maxgrid, sizeof(int) * maxgrid, maxgrid, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(mean_d, sizeof(int) * maxgrid, mean, sizeof(int) * maxgrid, sizeof(int) * maxgrid, maxgrid, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(path_d, sizeof(int) * maxgrid, path, sizeof(int) * maxgrid, sizeof(int) * maxgrid, maxgrid, cudaMemcpyHostToDevice));

    //kernel_reg_detect(niter, maxgrid, length, sum_tang_d, mean_d, path_d, diff_d, sum_diff_d);
    print_array(maxgrid, path);



    free(sum_tang);
    free(mean);
    free(path);

    cudaFree(sum_tang_d);
    cudaFree(mean_d);
    cudaFree(path_d);
    cudaFree(diff_d);
    cudaFree(sum_diff_d);
    
    
}
