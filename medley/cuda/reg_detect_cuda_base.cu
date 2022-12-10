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

static void init_array(int maxgrid, int** sum_tang, int** mean, int** path)
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++)
    {
      sum_tang[i][j] = (int)((i + 1) * (j + 1));
      mean[i][j] = ((int)i - j) / maxgrid;
      path[i][j] = ((int)i * (j - 1)) / maxgrid;
    }
}

static void print_array(int maxgrid, int** path)
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++)
    {
      fprintf(stdout, "%d ", path[i][j]);
      if ((i * maxgrid + j) % 20 == 0)
        fprintf(stdout, "\n");
    }
  fprintf(stdout, "\n");
}

static void kernel_reg_detect(int niter, int maxgrid, int length,int** sum_tang,int**mean,
                              int** path, int*** diff, int*** sum_diff)
{
  clock_t begin = clock();
  int t, i, j, cnt;

  for (t = 0; t < niter; t++)
  { 
    for (j = 0; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        for (cnt = 0; cnt <= length - 1; cnt++)
          diff[j][i][cnt] = sum_tang[j][i];  

    for (j = 0; j <= maxgrid - 1; j++)
    {
      for (i = j; i <= maxgrid - 1; i++) {
        sum_diff[j][i][0] = diff[j][i][0];
      }
    }

    for (j = 0; j <= maxgrid - 1; j++)
    {
      for (i = j; i <= maxgrid - 1; i++)
      {
        //sum_diff[j][i][0] = diff[j][i][0];

        for (cnt = 1; cnt <= length - 1; cnt++)
          sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
        
      }
    }

    for (j = 0; j <= maxgrid - 1; j++)
    {
      for (i = j; i <= maxgrid - 1; i++) {
        mean[j][i] = sum_diff[j][i][length - 1];
      }
    }

    for (i = 0; i <= maxgrid - 1; i++)
      path[0][i] = mean[0][i];

    for (j = 1; j <= maxgrid - 1; j++)
      for (i = j; i <= maxgrid - 1; i++)
        path[j][i] = path[j - 1][i - 1] + mean[j][i];
  }

  clock_t end = clock();
  printf("Elapsed time with custom timer: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

int main() {
    int niter = NITER;
    int maxgrid = MAXGRID;
    int length = LENGTH;

    int** sum_tang = (int**)malloc(sizeof(int*) * maxgrid);
    int** mean = (int**)malloc(sizeof(int*) * maxgrid);
    int** path = (int**)malloc(sizeof(int*) * maxgrid);
    int*** diff = (int***)malloc(sizeof(int**) * maxgrid);
    int*** sum_diff = (int***)malloc(sizeof(int**) * maxgrid);

    for(int i = 0; i < maxgrid; i++) {
        sum_tang[i] = (int*)malloc(sizeof(int) * maxgrid);
        mean[i] = (int*)malloc(sizeof(int) * maxgrid);
        path[i] = (int*)malloc(sizeof(int) * maxgrid);
        diff[i] = (int**)malloc(sizeof(int*) * maxgrid);
        sum_diff[i] = (int**)malloc(sizeof(int*) * maxgrid);
        for (int j = 0; j < maxgrid; j++) {
            diff[i][j] = (int*)malloc(sizeof(int) * length);
            sum_diff[i][j] = (int*)malloc(sizeof(int) * length);
        }

    }


    print_array(maxgrid, path);
    init_array(maxgrid, sum_tang, mean, path);
    kernel_reg_detect(niter, maxgrid, length, sum_tang, mean, path, diff, sum_diff);
    print_array(maxgrid, path);



    for(int i = 0; i < maxgrid; i++) {
        free(sum_tang[i]);
        free(mean[i]);
        free(path[i]);
        for (int j = 0; j < maxgrid; j++) {
            free(diff[i][j]);
            free(sum_diff[i][j]);
        }
        free(diff[i]);
        free(sum_diff[i]);
    }

    free(sum_tang);
    free(mean);
    free(path);
    free(diff);
    free(sum_diff);
    
    
}
