
void kernel(float *a)
__global__ void kernel( * a, * b, * c,  row,  col, indexA, indexB, indexC)
{
  int j;
  int k;
  int sum;
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
 
  if(i <row){
            for (j = 1; j<=dim; j++) {
if(j<col)
                int numNeighbors = getNeighbors(grid, i, j);
if(j<col)
                if ((grid[i][j] == 1) && (numNeighbors < 2))
                    newGrid[i][j] = 0;
                else if ((grid[i][j] == 1) && ((numNeighbors == 2) || (numNeighbors == 3)))
                    newGrid[i][j] = 1;
                else if ((grid[i][j] == 1) && (numNeighbors > 3))
                    newGrid[i][j] = 0;
                else if ((grid[i][j] == 0) && (numNeighbors == 3))
                    newGrid[i][j] = 1;
                else
                    newGrid[i][j] = grid[i][j];
            }
        }
}
#include <stdio.h>
#include <stdlib.h>
#define SRAND_VALUE 1985

int getNeighbors(int **grid,int i,int j)
{
  int numNeighbors;
//upper lower
  numNeighbors = (((((((grid[i + 1][j] + grid[i - 1][j]) + grid[i][j + 1]) + grid[i][j - 1]) + grid[i + 1][j + 1]) + grid[i - 1][j - 1]) + grid[i - 1][j + 1]) + grid[i + 1][j - 1]);
//right left
//diagonals
  return numNeighbors;
}

int main(int argc,char *argv[])
{
  int i;
  int j;
  int iter;
  int dim = 1024;
  int maxIter = 1 << 10;
  int **grid = (int **)(malloc((sizeof(int *) * (dim + 2))));
  for (i = 0; i < (dim + 2); i++) 
    grid[i] = ((int *)(malloc((sizeof(int *) * (dim + 2)))));
  int **newGrid = (int **)(malloc((sizeof(int *) * (dim + 2))));
  for (i = 0; i < (dim + 2); i++) 
    newGrid[i] = ((int *)(malloc((sizeof(int *) * (dim + 2)))));
  srand(1985);
  for (i = 1; i <= dim; i++) {
    for (j = 1; j <= dim; j++) {
      grid[i][j] = (rand() % 2);
    }
  }
  for (iter = 0; iter < maxIter; iter++) {
    for (i = 1; i <= dim; i++) {
      grid[i][0] = grid[i][dim];
      grid[i][dim + 1] = grid[i][1];
    }
    for (j = 0; j <= (dim + 1); j++) {
      grid[0][j] = grid[dim][j];
      grid[dim + 1][j] = grid[1][j];
    }
 
  /***** Starting Parallalization *****/
  //declare device variables
  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
 *  device_a;
 *  device_b;
 *  device_c;
 
  //Allocate memory space in the GPU
  cudaMalloc((void **) &device_a, sizeof(a));
  cudaMalloc((void **) &device_b, sizeof(b));
  cudaMalloc((void **) &device_c, sizeof(c));
 
  //Copy from host to device
  cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, sizeof(b), cudaMemcpyHostToDevice);
 
  //launch kernel function
  dim3 numThreads(32,32);
  dim3 blocks((row+ 31)/32, (col+ 31)/32);
  cudaEventRecord(start, 0);
  kernel<<<blocks,numThreads>>>(device_a,device_b,device_c, row, col,indexA,indexB,indexC);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("the elapsed time is %f\n", elapsedTime);
 
  //copy back from device to host 
  cudaMemcpy(c, device_c, sizeof(c), cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
 
  /***** Ending Parallalization *****/
    int **tmpGrid = grid;
    grid = newGrid;
    newGrid = tmpGrid;
  }
  int total = 0;
  for (i = 1; i <= dim; i++) {
    for (j = 1; j <= dim; j++) {
      total += grid[i][j];
    }
  }
  printf("Total Alive: %d\n",total);
  free(grid);
  free(newGrid);
  return 0;
}
