
void __global__ kernel(int *grid,int *newGrid,int dim,int maxIter)
{
  int j;
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
 
if ( i <=dim && j <=dim){
int numNeighbors =(((((((grid[(i+1)*(dim+2)+(j)]+grid[(i-1)*(dim+2)+(j)])+grid[(i)*(dim+2)+(j+1)])+grid[(i)*(dim+2)+(j-1)])+grid[(i+1)*(dim+2)+(j+1)])+grid[(i-1)*(dim+2)+(j-1)])+grid[(i-1)*(dim+2)+(j+1)])+grid[(i+1)*(dim+2)+(j-1)]);
if ((grid[(i)*(dim+2)+(j)] == 1) && (numNeighbors < 2))
newGrid[(i)*(dim+2)+(j)] = 0;
else if ((grid[(i)*(dim+2)+(j)] == 1) && ((numNeighbors == 2) || (numNeighbors == 3)))
newGrid[(i)*(dim+2)+(j)] = 1;
else if ((grid[(i)*(dim+2)+(j)] == 1) && (numNeighbors > 3))
newGrid[(i)*(dim+2)+(j)] = 0;
else if ((grid[(i)*(dim+2)+(j)] == 0) && (numNeighbors == 3))
newGrid[(i)*(dim+2)+(j)] = 1;
else
newGrid[(i)*(dim+2)+(j)] = grid[(i)*(dim+2)+(j)];
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
  int dim = 10;
  int maxIter = 1 << 4;
  int **grid = (int **)(malloc((sizeof(int *) * (dim + 2))));
  for (i = 0; i < (dim + 2); i++) 
    grid[i] = ((int *)(malloc((sizeof(int *) * (dim + 2)))));
  int **newGrid = (int **)(malloc((sizeof(int *) * (dim + 2))));
  for (i = 0; i < (dim + 2); i++) 
    newGrid[i] = ((int *)(malloc((sizeof(int *) * (dim + 2)))));
  srand(1985);
  for (i = 1; i <= dim; i++) {
    for (j = 1; j <= dim; j++) {
      grid[i][j] = ((2 * i) + (j * 2));
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
int *  device_grid;
int *  device_newGrid;
 
  //Allocate memory space in the GPU
int grid_flat[(dim+2)*(dim+2)];
int newGrid_flat[(dim+2)*(dim+2)];
for(int ii=0;ii<dim+2;ii++){for(int jj=0;jj<dim+2;jj++){grid_flat[ii*(dim+2)+jj] = grid[ii][jj];}}
for(int ii=0;ii<dim+2;ii++){for(int jj=0;jj<dim+2;jj++){newGrid_flat[ii*(dim+2)+jj] = newGrid[ii][jj];}}
  cudaMalloc((void **) &device_grid, sizeof(grid_flat));
  cudaMalloc((void **) &device_newGrid, sizeof(newGrid_flat));
 
  //Copy from host to device
  cudaMemcpy(device_grid, grid_flat, sizeof(grid_flat), cudaMemcpyHostToDevice);
  cudaMemcpy(device_newGrid, newGrid_flat, sizeof(newGrid_flat), cudaMemcpyHostToDevice);
 
  //launch kernel function
  dim3 numThreads(2,2);
  dim3 blocks((dim+ 1)/2, (dim+ 1)/2);
  cudaEventRecord(start, 0);
  kernel<<<blocks,numThreads>>>(device_grid,device_newGrid, dim, maxIter);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("the elapsed time is %f\n", elapsedTime);
 
  //copy back from device to host 
  cudaMemcpy(grid_flat, device_grid, sizeof(grid_flat), cudaMemcpyDeviceToHost);
  cudaMemcpy(newGrid_flat, device_newGrid, sizeof(newGrid_flat), cudaMemcpyDeviceToHost);
  cudaFree(device_grid);
  cudaFree(device_newGrid);
for(int ii=0;ii<dim+2;ii++){for(int jj=0;jj<dim+2;jj++){grid[ii][jj]=grid_flat[ii*(dim+2)+jj];}}
for(int ii=0;ii<dim+2;ii++){for(int jj=0;jj<dim+2;jj++){newGrid[ii][jj]=newGrid_flat[ii*(dim+2)+jj];}}
 
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
