
void __global__ kernel(void **u,void **unew,void **f,int nx,int ny,void dx,void dy)
{
  int i;
  int j;
  j = blockIdx.x * blockDim.x + threadIdx.x;
  i = blockIdx.y * blockDim.y + threadIdx.y;
 
  if(j <=ny)
}
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <iostream>
# define NX 11
# define NY 11
using namespace std;
int main(int argc,char *argv[]);
double r8mat_rms(int nx,int ny,double a[11UL][11UL]);
void rhs(int nx,int ny,double f[11UL][11UL]);
void sweep(int nx,int ny,double dx,double dy,double f[11UL][11UL],double u[11UL][11UL],double unew[11UL][11UL]);
void timestamp();
double u_exact(double x,double y);
double uxxyy_exact(double x,double y);
/******************************************************************************/

int main(int argc,char *argv[])
/******************************************************************************/
/*
 *   Purpose:
 *
 *       MAIN is the main program for POISSON_SERIAL.
 *
 *         Discussion:
 *
 *             POISSON_SERIAL is a program for solving the Poisson problem.
 *
 *                 This program runs serially.  Its output is used as a benchmark for
 *                     comparison with similar programs run in a parallel environment.
 *
 *                         The Poisson equation
 *
 *                               - DEL^2 U(X,Y) = F(X,Y)
 *
 *                                   is solved on the unit square [0,1] x [0,1] using a grid of NX by
 *                                       NX evenly spaced points.  The first and last points in each direction
 *                                           are boundary points.
 *
 *                                               The boundary conditions and F are set so that the exact solution is
 *
 *                                                     U(x,y) = sin ( pi * x * y )
 *
 *                                                         so that
 *
 *                                                               - DEL^2 U(x,y) = pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )
 *
 *                                                                   The Jacobi iteration is repeatedly applied until convergence is detected.
 *
 *                                                                       For convenience in writing the discretized equations, we assume that NX = NY.
 *
 *                                                                         Licensing:
 *
 *                                                                             This code is distributed under the GNU LGPL license. 
 *
 *                                                                               Modified:
 *
 *                                                                                   16 October 2011
 *
 *                                                                                     Author:
 *
 *                                                                                         John Burkardt
 *                                                                                         */
{
  int converged;
  double diff;
  double dx;
  double dy;
  double error;
  double f[11UL][11UL];
  int i;
  int it;
  int it_max = 1000;
  int j;
  int nx = 11;
  int ny = 11;
  double tolerance = 0.000001;
  double u[11UL][11UL];
  double u_norm;
  double udiff[11UL][11UL];
  double uexact[11UL][11UL];
  double unew[11UL][11UL];
  double unew_norm;
  double x;
  double y;
  dx = (1.0 / ((double )(nx - 1)));
  dy = (1.0 / ((double )(ny - 1)));
/*
 *   Print a message.
 *   */
  timestamp();
  printf("\n");
  printf("POISSON_SERIAL:\n");
  printf("  C version\n");
  printf("  A program for solving the Poisson equation.\n");
  printf("\n");
  printf("  -DEL^2 U = F(X,Y)\n");
  printf("\n");
  printf("  on the rectangle 0 <= X <= 1, 0 <= Y <= 1.\n");
  printf("\n");
  printf("  F(X,Y) = pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )\n");
  printf("\n");
  printf("  The number of interior X grid points is %d\n",nx);
  printf("  The number of interior Y grid points is %d\n",ny);
  printf("  The X grid spacing is %f\n",dx);
  printf("  The Y grid spacing is %f\n",dy);
/*
 *   Initialize the data.
 *   */
  rhs(nx,ny,f);
/*
 *   Set the initial solution estimate.
 *     We are "allowed" to pick up the boundary conditions exactly.
 *     */
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      if ((((i == 0) || (i == (nx - 1))) || (j == 0)) || (j == (ny - 1))) {
        unew[i][j] = f[i][j];
      }
      else {
        unew[i][j] = 0.0;
      }
    }
  }
  unew_norm = r8mat_rms(nx,ny,unew);
/*
 *   Set up the exact solution.
 *   */
  for (j = 0; j < ny; j++) {
    y = (((double )j) / ((double )(ny - 1)));
    for (i = 0; i < nx; i++) {
      x = (((double )i) / ((double )(nx - 1)));
      uexact[i][j] = u_exact(x,y);
    }
  }
  u_norm = r8mat_rms(nx,ny,uexact);
  printf("  RMS of exact solution = %g\n",u_norm);
/*
 *   Do the iteration.
 *   */
  converged = 0;
  printf("\n");
  printf("  Step    ||Unew||     ||Unew-U||     ||Unew-Exact||\n");
  printf("\n");
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      udiff[i][j] = (unew[i][j] - uexact[i][j]);
    }
  }
  error = r8mat_rms(nx,ny,udiff);
  printf("  %4d  %14g                  %14g\n",0,unew_norm,error);
{
    for (it = 1; it <= it_max; it++) {
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          u[i][j] = unew[i][j];
        }
      }
/*
 *   UNEW is derived from U by one Jacobi step.
 *   */
      sweep(nx,ny,dx,dy,f,u,unew);
/*
 *   Check for convergence.
 *   */
      
      
      
      
      
          for ( j = 0; j < ny; j++ )
                    {
                                    for ( i = 0; i < nx; i++ )
                                                    {

                                                                        cout << "unew after sweep" << unew[i][j] << endl;;
                                                                                      }
                                              }


      
      u_norm = unew_norm;
      unew_norm = r8mat_rms(nx,ny,unew);
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          udiff[i][j] = (unew[i][j] - u[i][j]);
        }
      }
      diff = r8mat_rms(nx,ny,udiff);
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          udiff[i][j] = (unew[i][j] - uexact[i][j]);
        }
      }
      error = r8mat_rms(nx,ny,udiff);
      printf("  %4d  %14g  %14g  %14g\n",it,unew_norm,diff,error);
      if (diff <= tolerance) {
        converged = 1;
        break; 
      }
    }
  }
  if (converged != 0) {
    printf("  The iteration has converged.\n");
  }
  else {
    printf("  The iteration has NOT converged.\n");
  }
/*
 *   Terminate.
 *   */
  printf("\n");
  printf("POISSON_SERIAL:\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();
  return 0;
}
/******************************************************************************/

double r8mat_rms(int nx,int ny,double a[11UL][11UL])
/******************************************************************************/
/*
 *   Purpose:
 *
 *       R8MAT_RMS returns the RMS norm of a vector stored as a matrix.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license.
 *
 *               Modified:
 *
 *                   01 March 2003
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Input, int NX, NY, the number of rows and columns in A.
 *
 *                                   Input, double A[NX][NY], the vector.
 *
 *                                       Output, double R8MAT_RMS, the root mean square of the entries of A.
 *                                       */
{
  int i;
  int j;
  double v;
  v = 0.0;
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      v = (v + (a[i][j] * a[i][j]));
    }
  }
  v = sqrt((v / ((double )(nx * ny))));
  return v;
}
/******************************************************************************/

void rhs(int nx,int ny,double f[11UL][11UL])
/******************************************************************************/
/*
 *   Purpose:
 *
 *       RHS initializes the right hand side "vector".
 *
 *         Discussion:
 *
 *             It is convenient for us to set up RHS as a 2D array.  However, each
 *                 entry of RHS is really the right hand side of a linear system of the
 *                     form
 *
 *                           A * U = F
 *
 *                               In cases where U(I,J) is a boundary value, then the equation is simply
 *
 *                                     U(I,J) = F(i,j)
 *
 *                                         and F(I,J) holds the boundary data.
 *
 *                                             Otherwise, the equation has the form
 *
 *                                                   (1/DX^2) * ( U(I+1,J)+U(I-1,J)+U(I,J-1)+U(I,J+1)-4*U(I,J) ) = F(I,J)
 *
 *                                                       where DX is the spacing and F(I,J) is the value at X(I), Y(J) of
 *
 *                                                             pi^2 * ( x^2 + y^2 ) * sin ( pi * x * y )
 *
 *                                                               Licensing:
 *
 *                                                                   This code is distributed under the GNU LGPL license. 
 *
 *                                                                     Modified:
 *
 *                                                                         28 October 2011
 *
 *                                                                           Author:
 *
 *                                                                               John Burkardt
 *
 *                                                                                 Parameters:
 *
 *                                                                                     Input, int NX, NY, the X and Y grid dimensions.
 *
 *                                                                                         Output, double F[NX][NY], the initialized right hand side data.
 *                                                                                         */
{
  double fnorm;
  int i;
  int j;
  double x;
  double y;
/*
 *   The "boundary" entries of F store the boundary values of the solution.
 *     The "interior" entries of F store the right hand sides of the Poisson equation.
 *     */
  for (j = 0; j < ny; j++) {
    y = (((double )j) / ((double )(ny - 1)));
    for (i = 0; i < nx; i++) {
      x = (((double )i) / ((double )(nx - 1)));
      if ((((i == 0) || (i == (nx - 1))) || (j == 0)) || (j == (ny - 1))) {
        f[i][j] = u_exact(x,y);
      }
      else {
        f[i][j] = -uxxyy_exact(x,y);
      }
    }
  }
  fnorm = r8mat_rms(nx,ny,f);
  printf("  RMS of F = %g\n",fnorm);
}
/******************************************************************************/

void sweep(int nx,int ny,double dx,double dy,double f[11UL][11UL],double u[11UL][11UL],double unew[11UL][11UL])
/******************************************************************************/
/*
 *   Purpose:
 *
 *      SWEEP carries out one step of the Jacobi iteration.
 *
 *        Discussion:
 *
 *            Assuming DX = DY, we can approximate
 *
 *                  - ( d/dx d/dx + d/dy d/dy ) U(X,Y) 
 *
 *                      by
 *
 *                            ( U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) - 4*U(i,j) ) / dx / dy
 *
 *                                The discretization employed below will not be correct in the general
 *                                    case where DX and DY are not equal.  It's only a little more complicated
 *                                        to allow DX and DY to be different, but we're not going to worry about 
 *                                            that right now.
 *
 *                                              Licensing:
 *
 *                                                  This code is distributed under the GNU LGPL license. 
 *
 *                                                    Modified:
 *
 *                                                        26 October 2011
 *
 *                                                          Author:
 *
 *                                                              John Burkardt
 *
 *                                                                Parameters:
 *
 *                                                                    Input, int NX, NY, the X and Y grid dimensions.
 *
 *                                                                        Input, double DX, DY, the spacing between grid points.
 *
 *                                                                            Input, double F[NX][NY], the right hand side data.
 *
 *                                                                                Input, double U[NX][NY], the previous solution estimate.
 *
 *                                                                                    Output, double UNEW[NX][NY], the updated solution estimate.
 *                                                                                    */
{
  int i;
  int j;
 
  /***** Starting Parallalization *****/
  //declare device variables
  float elapsedTime;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
 **  device_u;
 **  device_unew;
 **  device_f;
 
  //Allocate memory space in the GPU
  cudaMalloc((void **) &device_u, sizeof(u));
  cudaMalloc((void **) &device_unew, sizeof(unew));
  cudaMalloc((void **) &device_f, sizeof(f));
 
  //Copy from host to device
  cudaMemcpy(device_u, u, sizeof(u), cudaMemcpyHostToDevice);
  cudaMemcpy(device_unew, unew, sizeof(unew), cudaMemcpyHostToDevice);
 
  //launch kernel function
  dim3 numThreads(32,32);
  dim3 blocks((ny+ 31)/32, (nx+ 31)/32);
  cudaEventRecord(start, 0);
  kernel<<<blocks,numThreads>>>(device_u,device_unew,device_f, nx, ny,dx,dy);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("the elapsed time is %f\n", elapsedTime);
 
  //copy back from device to host 
  cudaMemcpy(f, device_f, sizeof(f), cudaMemcpyDeviceToHost);
  cudaFree(device_u);
  cudaFree(device_unew);
  cudaFree(device_f);
 
  /***** Ending Parallalization *****/
}
/******************************************************************************/

void timestamp()
/******************************************************************************/
/*
 *   Purpose:
 *
 *       TIMESTAMP prints the current YMDHMS date as a time stamp.
 *
 *         Example:
 *
 *             31 May 2001 09:45:54 AM
 *
 *               Licensing:
 *
 *                   This code is distributed under the GNU LGPL license. 
 *
 *                     Modified:
 *
 *                         24 September 2003
 *
 *                           Author:
 *
 *                               John Burkardt
 *
 *                                 Parameters:
 *
 *                                     None
 *                                     */
{
# define TIME_SIZE 40
  static char time_buffer[40UL];
  const struct tm *tm;
  time_t now;
  now = time(0);
  tm = (localtime((&now)));
  strftime(time_buffer,40,"%d %B %Y %I:%M:%S %p",tm);
  printf("%s\n",time_buffer);
# undef TIME_SIZE
}
/******************************************************************************/

double u_exact(double x,double y)
/******************************************************************************/
/*
 *   Purpose:
 *
 *       U_EXACT evaluates the exact solution.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license.
 *
 *               Modified:
 *
 *                   25 October 2011
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Input, double X, Y, the coordinates of a point.
 *
 *                                   Output, double U_EXACT, the value of the exact solution 
 *                                       at (X,Y).
 *                                       */
{
  double pi = 3.141592653589793;
  double value;
  value = sin(((pi * x) * y));
  return value;
}
/******************************************************************************/

double uxxyy_exact(double x,double y)
/******************************************************************************/
/*
 *   Purpose:
 *
 *       UXXYY_EXACT evaluates ( d/dx d/dx + d/dy d/dy ) of the exact solution.
 *
 *         Licensing:
 *
 *             This code is distributed under the GNU LGPL license.
 *
 *               Modified:
 *
 *                   25 October 2011
 *
 *                     Author:
 *
 *                         John Burkardt
 *
 *                           Parameters:
 *
 *                               Input, double X, Y, the coordinates of a point.
 *
 *                                   Output, double UXXYY_EXACT, the value of 
 *                                       ( d/dx d/dx + d/dy d/dy ) of the exact solution at (X,Y).
 *                                       */
{
  double pi = 3.141592653589793;
  double value;
  value = (((-pi * pi) * ((x * x) + (y * y))) * sin(((pi * x) * y)));
  return value;
}
# undef NX
# undef NY
