/*
copied from tomo.f (Feb 92)
c     For each grid point, use the dusum and nrays from cells in a
c       (ism)X(jsm)X(ksm) area. This smooths du, but does not use the
c       zeroes in the unconstrained cells in the smoothing. Note that a
c       ray closer to the grid point will cross more cells, and will thus
c       be weighted heavier than a more distant ray. Note that this smoothing
c       is weighted by number of rays, while the smoothing in duadd.f is
c       unweighted. Both smoothings have advantages.
c
cc      program tomo
ccc
ccc     Using the output from cover.f, find the model perturbation, du, at
ccc        the grid points. For each grid point, add the ray perturbations
ccc        for all 8 neighbouring cells, and divide by the number of terms 
ccc        (i.e. find the average du for all rays for all cells) Note that a
ccc        ray near the grid point will affect 2 to 4 cells, and will thus be
ccc        weighted heavier than a more distant ray. Within the source cubes
ccc        (5X5X5 grid points), give all grid points the value at the source.
ccc
ccc     Input:   model's nx,ny,nz,h,x0,y0,0
ccc              xs,ys,zs for each source used
ccc              nrays(i,j,k) number of rays in each cell
ccc              sumdu(i,j,k) sum of du from rays in each cell
ccc
ccc     Output:  du(i,j,k) velocity model perturbation at grid points
ccc
ccc     J.Hole   Mar 1991
ccc              Apr 1991 commented out part handling source cube
ccc
c     J.Hole   Feb 1992
c              Jul 1993 MAJOR CHANGE: allow a resampling for the inversion.
c                       This option divides the effective sampling interval
c                       of the inversion (but not the model!) by the given
c                       factors ireg,jreg,kreg. The arrays dus and nrays are
c                       resampled by summing over all small cells within the
c                       new big cells. This has the effect of weighting by
c                       the length of the ray, measured approximately, in each
c                       big cell. Subsequent smoothing (ism,jsm,ksm) and 
c                       creation of du(gridpoints) are at the new coarser
c                       grid interval. For input to duadd.f, nx,ny,nz are still
c                       defined WITHOUT resampling, and they and ireg,jreg,kreg
c                       should be identical to this code.
c              Feb 1994 change to properly allow 2d models
c                       change to use 1d arrays to ease limit problems
	c	  C. Chacon 2011
	* 			Ported de code to C. Cyber-ShARE Center of Excellence.
	* 			The University of Texas at El Paso
c
c     ****** replace all recl=4* (on Sun) with recl=1* (on DEC)

*/
#include    <stdio.h>
#define      NXYZMX 500000L //50000000L
#define min(x1,x2) ((x1) > (x2) ? (x2):(x1))

int main()
{
  int nx;
  int ny;
  int nz;
  int i;
  int j;
  int k;
  int nrtot;
  int ism;
  int jsm;
  int ksm;
  int iiii;
  int iiiiii;
  int imin;
  int imax;
  int jmin;
  int jmax;
  int kmin;
  int kmax;
  int ii;
  int jj;
  int kk;
  int ireg;
  int jreg;
  int kreg;
  int nxr;
  int nyr;
  int nzr;
  int ix2d;
  int iy2d;
  int iz2d;
  int nrays[500000UL];
  float du[500000UL];
  float dus[500000UL];
  char nrfile[80UL];
  char dufile[80UL];
  char dsfile[80UL];
//File objects for the inputs
  FILE *Fnrfile;
  FILE *Fdufile;
  FILE *Fdsfile;
  printf("Tomosm Cover (J.Hole) \n");
  printf("Input the number-of-rays coverage filename \n");
  scanf("%s",nrfile);
  printf("Input the sum-of-rays du filename \n");
  scanf("%s",dsfile);
  printf("Input the slowness model perturbation filename \n");
  scanf("%s",dufile);
  printf("Input nx,ny,nz,   max nx*ny*nz =%ld \n",500000L);
  scanf("%d %d %d",&nx,&ny,&nz);
  if (((nx * ny) * nz) > 500000L) {
    printf("ERROR:  dimensions are too big");
  }
  printf("Input i,j,k factors for coarser inversion sampling \n");
  printf(" 1,1,1 equals no resampling");
  scanf("%d %d %d",&ireg,&jreg,&kreg);
  if (((ireg < 1) || (jreg < 1)) || (kreg < 1)) {
    printf("ERROR:  Invalid Size for Regridding");
  }
  printf("Input the i,j,k sizes of the smoothing volume \n");
  printf(" Each must be even and positive (i.e. 2,4,6...)\n");
  scanf("%d %d %d",&ism,&jsm,&ksm);
  if (((((((ism % 2) != 0) || ((jsm % 2) != 0)) || ((ksm % 2) != 0)) || (ism < 2)) || (jsm < 2)) || (ksm < 2)) {
    printf("ERROR:  Invalid Size for Smoothing Volume\n");
  }
  ism = (ism / 2);
  jsm = (jsm / 2);
  ksm = (ksm / 2);
  printf("testing for 2d models\n");
//c     TEST FOR 2D MODELS
  ix2d = 0;
  iy2d = 0;
  iz2d = 0;
  if (nx == 1) {
    nx = 2;
    ix2d = 1;
    printf("2d model encountered (nx=1)\n");
  }
  if (ny == 1) {
    ny = 2;
    iy2d = 1;
    printf("2d model encountered (ny=1)\n");
  }
  if (nz == 1) {
    nz = 2;
    iz2d = 1;
    printf("2d model encountered (nz=1)\n");
  }
  if (((nx * ny) * nz) > 500000L) {
    printf("***ERROR:  dimensions are too big (for 2d)\n");
  }
  printf("Opening 3D Files\n");
//c     OPEN AND READ 3D COVERAGE AND DU FILES
//open coverage file
  Fnrfile = fopen(nrfile,"rb");
//open dusum file
  Fdsfile = fopen(dsfile,"rb");
//*****************************************Load with for loops
  for (k = 1; k <= (nz - 1); k++) {
    for (j = 1; j <= (ny - 1); j++) {
      for (i = 1; i <= (nx - 1); i++) {
        fread((nrays + (((((nx - 1) * (ny - 1)) * (k - 1)) + ((nx - 1) * (j - 1))) + i)),(sizeof(int )),1,Fnrfile);
        fread((dus + (((((nx - 1) * (ny - 1)) * (k - 1)) + ((nx - 1) * (j - 1))) + i)),(sizeof(float )),1,Fdsfile);
      }
    }
  }
  fclose(Fnrfile);
  fclose(Fdsfile);
//c     REGRID DUS AND NRAYS TO LARGER GRID CELLS
  if (((ireg > 1) || (jreg > 1)) || (kreg > 1)) {
    nxr = ((nx - 1) / ireg);
    if ((nxr * ireg) != (nx - 1)) {
      printf("WARNING: resampling is uneven at max x\n");
      nxr = (nxr + 1);
    }
    nyr = ((ny - 1) / jreg);
    if ((nyr * jreg) != (ny - 1)) {
      printf("WARNING: resampling is uneven at max y\n");
      nyr = (nyr + 1);
    }
    nzr = ((nz - 1) / kreg);
    if ((nzr * kreg) != (nz - 1)) {
      printf("WARNING: resampling is uneven at max z\n");
      nzr = (nzr + 1);
    }
    printf("new number of grid CELLS =  %d  %d  %d",nxr,nyr,nzr);
    for (k = 1; k <= nzr; k++) {
      for (j = 1; j <= nyr; j++) {
        for (i = 1; i <= nxr; i++) {
          iiii = ((((nxr * nyr) * (k - 1)) + (nxr * (j - 1))) + i);
          if (((i != 1) || (j != 1)) || (k != 1)) {
//OK TO OVERWRITE ARRAYS BECAUSE ii>i, jj>j, kk>k, 
//nx>nxr, ny>nyr ( THEREFORE iiiiii>iiii )  *AND*
//BECAUSE ARRAYS ARE NESTED SAME AS DO-LOOPS
            dus[iiii] = 0.;
            nrays[iiii] = 0;
          }
          for (kk = (((k - 1) * kreg) + 1); kk <= ((((k * kreg) > (nz - 1))?(nz - 1) : (k * kreg))); kk++) {
            for (jj = (((j - 1) * jreg) + 1); jj <= ((((j * jreg) > (ny - 1))?(ny - 1) : (j * jreg))); jj++) {
              for (ii = (((i - 1) * ireg) + 1); ii <= ((((i * ireg) > (nx - 1))?(nx - 1) : (i * ireg))); ii++) {
                if (((ii != 1) || (jj != 1)) || (kk != 1)) {
//                   OK TO OVERWRITE ARRAYS BECAUSE ii>i, jj>j, kk>k,
//                   nx>nxr, ny>nyr ( THEREFORE iiiiii>iiii )  *AND*
//                   BECAUSE ARRAYS ARE NESTED SAME AS DO-LOOPS
                  iiiiii = (((((nx - 1) * (ny - 1)) * (kk - 1)) + ((nx - 1) * (jj - 1))) + ii);
                  dus[iiii] = (dus[iiii] + dus[iiiiii]);
                  nrays[iiii] = (nrays[iiii] + nrays[iiiiii]);
                }
              }
            }
          }
        }
      }
    }
//FOR THE REST OF THIS PROGRAM ONLY...
//(WHEN RUNNING duadd.f, USE THE OLD VALUES FOR nx,ny,nz)
    nx = (nxr + 1);
    ny = (nyr + 1);
    nz = (nzr + 1);
    printf("new number of grid NODES %d  %d  %d",nx,ny,nz);
  }
  printf("Find du at grid points\n");
//c     FIND DU AT GRID POINTS
  for (k = 1; k <= nz; k++) {
    for (j = 1; j <= ny; j++) {
      for (i = 1; i <= nx; i++) {
        imin = (i - ism);
        imax = ((i + ism) - 1);
        jmin = (j - jsm);
        jmax = ((j + jsm) - 1);
        kmin = (k - ksm);
        kmax = ((k + ksm) - 1);
        if (imin < 1) 
          imin = 1;
        if (imax > (nx - 1)) 
          imax = (nx - 1);
        if (jmin < 1) 
          jmin = 1;
        if (jmax > (ny - 1)) 
          jmax = (ny - 1);
        if (kmin < 1) 
          kmin = 1;
        if (kmax > (nz - 1)) 
          kmax = (nz - 1);
        iiii = ((((nx * ny) * (k - 1)) + (nx * (j - 1))) + i);
        du[iiii] = 0.;
        nrtot = 0;
        for (kk = kmin; kk <= kmax; kk++) {
          for (jj = jmin; jj <= jmax; jj++) {
            for (ii = imin; ii <= imax; ii++) {
              iiiiii = (((((nx - 1) * (ny - 1)) * (kk - 1)) + ((nx - 1) * (jj - 1))) + ii);
              du[iiii] = (du[iiii] + dus[iiiiii]);
              nrtot = (nrtot + nrays[iiiiii]);
            }
          }
        }
        if (nrtot != 0) {
          du[iiii] = (du[iiii] / ((float )nrtot));
        }
        else {
          du[iiii] = 0.;
        }
      }
    }
  }
  printf("nrtot: %d \n",nrtot);
  printf("Prepare to write 2d model correctly\n");
//	PREPARE TO WRITE 2D MODEL CORRECTLY
  if (ix2d == 1) {
    nx = 1;
    for (k = 1; k <= nz; k++) {
      for (j = 1; j <= ny; j++) {
        iiii = ((((nx * ny) * (k - 1)) + (nx * (j - 1))) + 1);
        du[iiii] = du[(((2 * ny) * (k - 1)) + (2 * (j - 1))) + 1];
      }
    }
  }
  if (iy2d == 1) {
    ny = 1;
    for (k = 1; k < nz; k++) {
      for (i = 1; i <= nx; i++) {
        iiii = (((nx * ny) * (k - 1)) + i);
        du[iiii] = du[((nx * 2) * (k - 1)) + i];
      }
    }
  }
  if (iz2d == 1) {
    nz = 1;
  }
  printf("Wite du file \n");
  Fdufile = fopen(dufile,"w");
  for (k = 1; k <= nz; k++) {
    for (j = 1; j <= ny; j++) {
      iiii = (((nx * ny) * (k - 1)) + (nx * (j - 1)));
      for (i = 1; i <= nx; i++) {
        fwrite((du + (iiii + i)),(sizeof(float )),1,Fdufile);
      }
    }
  }
  fclose(Fdufile);
  return 0;
}
