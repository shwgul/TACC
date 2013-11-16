#include <stdio.h>
#include <stdlib.h>

#define SRAND_VALUE 1985

int getNeighbors(int** grid, int i, int j)
{
  int numNeighbors;
  numNeighbors = grid[i+1][j] + grid[i-1][j]     //upper lower
    + grid[i][j+1] + grid[i][j-1]     //right left
    + grid[i+1][j+1] + grid[i-1][j-1] //diagonals
    + grid[i-1][j+1] + grid[i+1][j-1];

  return numNeighbors;
}

int main(int argc, char* argv[])
{
  int i,j,iter;
  int dim = 10;
  int maxIter = 1<<4;

  int **grid = (int**) malloc( sizeof(int*) * (dim+2) );
  for(i = 0; i<dim+2; i++)
    grid[i] = (int*) malloc( sizeof(int*) * (dim+2) );

  int **newGrid = (int**) malloc( sizeof(int*) * (dim+2) );
  for(i = 0; i<dim+2; i++)
    newGrid[i] = (int*) malloc( sizeof(int*) * (dim+2) );

  srand(SRAND_VALUE);
  for(i = 1; i<=dim; i++) {
    for(j = 1; j<=dim; j++) {
      grid[i][j] = 2*i+j*2;
    }
  }

  for (iter = 0; iter<maxIter; iter++) {
    for (i = 1; i<=dim; i++) {
      grid[i][0] = grid[i][dim]; 
      grid[i][dim+1] = grid[i][1];         
    }


    for (j = 0; j<=dim+1; j++) {
      grid[0][j] = grid[dim][j]; 
      grid[dim+1][j] = grid[1][j];
    } 

    for (i = 1; i<=dim; i++) {
      for (j = 1; j<=dim; j++) {

        int numNeighbors = grid[i+1][j] + grid[i-1][j] + grid[i][j+1] + grid[i][j-1]+ grid[i+1][j+1] + grid[i-1][j-1]+ grid[i-1][j+1] + grid[i+1][j-1];

        if (grid[i][j] == 1 && numNeighbors < 2)
          newGrid[i][j] = 0;
        else if (grid[i][j] == 1 && (numNeighbors == 2 || numNeighbors == 3))
          newGrid[i][j] = 1;
        else if (grid[i][j] == 1 && numNeighbors > 3)
          newGrid[i][j] = 0;
        else if (grid[i][j] == 0 && numNeighbors == 3)
          newGrid[i][j] = 1;
        else
          newGrid[i][j] = grid[i][j];
      }
    }

    int **tmpGrid = grid;
    grid = newGrid;
    newGrid = tmpGrid;
  }

  int total = 0;
  for (i = 1; i<=dim; i++) {
    for (j = 1; j<=dim; j++) {
      total += grid[i][j];
    }
  }

  printf("Total Alive: %d\n", total);
  free(grid);
  free(newGrid); 
  return 0;
}
