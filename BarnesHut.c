/* =======================================================================
 *    nbody.c
 *
 *       DESCRIPTION:
 *
 *            This program computes the interaction that N bodies exert on
 *                 each other in 2-D space.  Each body is defined by its (x,y) position
 *                      and its mass.
 *
 *                           The program simulates the interaction over some number of steps
 *                                provided by the user (NoSteps).
 *
 *                                   SYNTAX:
 *                                           ld-net nbody  NumberOfBodies NumberOfTransputers NumberOfSteps
 *
 *                                              VIRTUAL network:
 *
 *                                                       +-----+   +-----+   +-----+   +-----+             +-----+
 *                                                           +----|  1  +---|  2  +---|  3  +---|  4  +---  . . . --|  P  +----+
 *                                                               |    +-----+   +-----+   +-----+   +-----+             +-----+    |
 *                                                                   |                                                                 |
 *                                                                       +-----------------------------------------------------------------+
 *
 *                                                                            Example with 6 transputers
 *                                                                                 Node[Vchan] ---> Node[Vchan]
 *                                                                                      --------------------------
 *                                                                                                1[6] --->2[12]
 *                                                                                                          2[7] --->3[13]
 *                                                                                                                    3[8] --->4[14]
 *                                                                                                                              4[9] --->5[15]
 *                                                                                                                                        5[10]--->6[16]
 *                                                                                                                                                  6[11]--->1[17]
 *
 *                                                                                                                                                   ====================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "conc.h"

#define G           6.670E-11           /* gravitational constant       */
#define sqr(x)      ((x)*(x))
#define swap(x,y,z) z=x; x=y; y=z
#define BODYSIZE    ((int) sizeof(BODY))
#define FORCESIZE   ((int) sizeof(FORCE))

typedef struct body BODY;
struct body
{
      float x;           /* coordinates */
          float y;
              float mass;
};

typedef struct force FORCE;
struct force
{
      float x;           /* x direction of force */
          float y;           /* y direction of force */
};

/* ============================= GLOBALS ============================== */
BODY  *MyBodies;    /* array of all bodies maintained by this node      */
FORCE *MyForces;    /* array of forces acting on them                   */
BODY  *OtherBodies1;/* array of bodies passed from node to node         */
BODY  *OtherBodies2;/* needs a second one when we receive the neighbor's*/
                    /* copy before we can send out our own.             */
BODY  *TempBodies;  /* used for swapping OtherBodies arrays             */
FORCE *OtherForces; /* array of forces for those bodies                 */
int   P;            /* number of transputers in network                 */
int   N;            /* total number of bodies in system                 */
int   MyN;          /* number of bodies maintained by this node         */
int   NoSteps;      /* total number of iterations required              */
Channel *left;      /* virtual channel with left neighbor               */
Channel *right;     /* virtual channel with right neighbor              */

/* =========================== PROTOTYPES ============================= */
void Initialize(void);
void ComputeForces(void);
void DoComputation(void);

/* -------------------------------------------------------------------- */
/*                                 MAIN                                 */
/* -------------------------------------------------------------------- */
void main(int argc, char *argv[])
{
      if (argc<4)
            {
                      if (_node_number==1)
                                    fprintf(stderr,"%s%s",
                                                                    "Syntax: ld-net nbody #bodies ",
                                                                                                   "#Nodes #simulation_steps\n");
                              exit(1);
                                  }

          _heapend = (void *)0x800FFFFC;  /* maximize heap size */

              N       = atoi(argv[1]);
                  P       = atoi(argv[2]);
                      NoSteps = atoi(argv[3]);

                          Initialize();
                              DoComputation();
                                  exit(0);
}

/* -------------------------------------------------------------------- */
/* INITIALIZE                                                           */
/* Allocates arrays for private and temporary slices                    */
/* Creates left and right virtual channels                              */
/* -------------------------------------------------------------------- */
void Initialize()
{
      int i;

          MyN            = N/P;

              /*--- allocate storage ---*/
              MyBodies     = (BODY *)  malloc((size_t) N*BODYSIZE);
                  MyForces     = (FORCE *) malloc((size_t) N*FORCESIZE);
                      OtherBodies1 = (BODY *)  malloc((size_t) N*BODYSIZE);
                          OtherBodies2 = (BODY *)  malloc((size_t) N*BODYSIZE);
                              OtherForces  = (FORCE *) malloc((size_t) N*FORCESIZE);

                                  if ((!MyBodies) || (!MyForces) || (!OtherBodies1)
                                                || (!OtherBodies2) || (!OtherForces))
                                        {
                                                  if (_node_number==1)
                                                                fprintf(stderr,"malloc error\n");
                                                          exit(1);
                                                              }

                                      /*--- initialize bodies maintained by this node ---*/
                                      for (i=0; i<MyN; i++)
                                            {
                                                      MyBodies[i].x    = rand()*1.0;
                                                              MyBodies[i].y    = rand()*1.0;
                                                                      MyBodies[i].mass = rand()*1.0;
                                                                          }

                                          /*--- initialize virtual channels ---*/
                                          right = VChan(5+_node_number);
                                              left        = (_node_number==1)? VChan(5+2*P)
                                                                                : VChan(4+P+_node_number);
}

/* -------------------------------------------------------------------- */
/* CLEARFORCES                                                          */
/* -------------------------------------------------------------------- */
void ClearForces(void)
{
      int i;

          for (i=0; i<MyN; I++)
                {
                          MyForces[i].x = 0;
                                  MyForces[i].y = 0;
                                      }
}

/* -------------------------------------------------------------------- */
/* DOCOMPUTATION                                                        */
/* Takes care of the simulation steps.  Each step start with the nodes  */
/* sending MyBodies to the right neighbor.  Then the cycle of (Compute, */
/* Shift) operations start.  This program does not do anything with     */
/* the resulting forces, but discards them.                             */
/* -------------------------------------------------------------------- */
void DoComputation()
{
      int i, step;

          printf("%d started\n",_node_number);

              /*--- do all simulation steps---*/
              for (step = 0; step < NoSteps; step++)
                    {
                              /*--- clear the forces ---*/
                              ClearForces();

                                      /*--- First send our bodies to right neighbor ---*/
                                      if (_node_number%2) /* odd Ids send first*/
                                                {
                                                              VChanOut(right,MyBodies,MyN*BODYSIZE);
                                                                          VChanIn(left,OtherBodies1,MyN*BODYSIZE);
                                                                                  }
                                              else                /* even Ids receive first */
                                                        {
                                                                      VChanIn(left,OtherBodies1,MyN*BODYSIZE);
                                                                                  VChanOut(right,MyBodies, MyN*BODYSIZE);
                                                                                          }

                                                      /*--- Then we compute and shift the bodies ---*/
                                                      /*--- so that we go once around the ring.  ---*/
                                                      for (i=0; i<P-2; i++)
                                                                {
                                                                              ComputeForces();
                                                                                          if (_node_number%2) /* Odd Ids send first */
                                                                                                        {
                                                                                                                          VChanOut(right,OtherBodies1,MyN*BODYSIZE);
                                                                                                                                          VChanIn(left,  OtherBodies1,MyN*BODYSIZE);
                                                                                                                                                      }
                                                                                                      else                /* Even Ids receive first */
                                                                                                                    {
                                                                                                                                      VChanIn(left,OtherBodies2, MyN*BODYSIZE);
                                                                                                                                                      VChanOut(right,OtherBodies1, MyN*BODYSIZE);
                                                                                                                                                                      /*---swap OtherBodies arrays so that OtherBodies1 ---*/
                                                                                                                                                                      /*---contains the new slice                       ---*/
                                                                                                                                                                      swap(OtherBodies1, OtherBodies2, TempBodies);
                                                                                                                                                                                  }
                                                                                                              }
                                                          }
}

/* -------------------------------------------------------------------- */
/* COMPUTEFORCES                                                        */
/* Computes the gravitational force between private slice               */
/* (MyBodies) and temporary slice (OtherBodies1).                       */
/* -------------------------------------------------------------------- */
void ComputeForces()
{
      int   i, j;
          float distance2;                   /* Square of distance            */
              float distance;                    /* distance between two bodies   */
                  float distancex;                   /* x component of distance       */
                      float distancey;                   /* y component of distance       */
                          float ReciprocalForce;             /* force between two bodies      */

                              for (i=0; i<MyN; i++)
                                    {
                                              for (j=0; j<MyN; j++)
                                                        {
                                                                      if (i==j) continue;
                                                                                  distancex = MyBodies[i].x-OtherBodies1[j].x;
                                                                                              distancey = MyBodies[i].y-OtherBodies1[j].y;
                                                                                                          distance2 = sqr(distancex) + sqr(distancey);
                                                                                                                      distance  = sqrt((double) distance2);
                                                                                                                                  ReciprocalForce= G * MyBodies[i].mass * OtherBodies1[j].mass
                                                                                                                                                                     / distance2;
                                                                                                                                              MyForces[i].x += ReciprocalForce * distancex/distance;
                                                                                                                                                          MyForces[i].y += ReciprocalForce * distancey/distance;
                                                                                                                                                                  }
                                                  }
}

