/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints, 	     		/* total points along string */
    rcode;                  	/* generic return code */
float  values[MAXPOINTS+2], 	/* values at time t */
       oldval[MAXPOINTS+2], 	/* values at time (t-dt) */
       newval[MAXPOINTS+2]; 	/* values at time (t+dt) */


int maxThreadsPerBlock;
float* valptr;
float* oldptr;
float* newptr;
int* datasizeptr;
float cuda_val[MAXPOINTS+2];

__global__ void cuda_do_math(float* values, float* oldval, float* newval, int* datasize) {
	
	float dtime, c, dx, tau, sqtau;
	dtime = 0.3;
	c = 1.0;
	dx = 1.0;
	tau= (c * dtime / dx);
	sqtau = tau * tau;
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < (*datasize)) {
	    newval[i] = (2.0 * values[i]) - oldval[i] + (sqtau *  (-2.0) * values[i]);
	}
}


/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

/**********************************************************************
 *     Initialize points on line
 *********************************************************************/
void init_line(void)
{
   int i, j;
   float x, fac, k, tmp;

   /* Calculate initial values based on sine curve */
   fac = 2.0 * PI;
   k = 0.0; 
   tmp = tpoints - 1;
   for (j = 1; j <= tpoints; j++) {
      x = k/tmp;
      values[j] = sin (fac * x);
      k = k + 1.0;
   } 

   /* Initialize old values array */
   for (i = 1; i <= tpoints; i++) 
      oldval[i] = values[i];
}

/**********************************************************************
 *      Calculate new values using wave equation
 *********************************************************************/
void do_math(int i)
{
   float dtime, c, dx, tau, sqtau;

   dtime = 0.3;
   c = 1.0;
   dx = 1.0;
   tau = (c * dtime / dx);
   sqtau = tau * tau;
   newval[i] = (2.0 * values[i]) - oldval[i] + (sqtau *  (-2.0)*values[i]);
}

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
void update()
{
   int i, j;

   int numBlocks = tpoints / maxThreadsPerBlock;
   if (tpoints % maxThreadsPerBlock != 0) {
   	numBlocks++;
   }
   /* Update values for each time step */
   for (i = 1; i<= nsteps; i++) {
      
      cuda_do_math<<<numBlocks, maxThreadsPerBlock>>>(valptr, oldptr, newptr, datasizeptr);
      float* temp;
      temp = oldptr;
      oldptr = valptr;
      valptr = newptr;
      newptr = temp; 
      /* Update old values with new values */
      for (j = 1; j <= tpoints; j++) {
         oldval[j] = values[j];
         values[j] = newval[j];
      }
   }
   cudaMemcpy(cuda_val, valptr, sizeof(float) * tpoints, cudaMemcpyDeviceToHost); 
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal_cuda()
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", cuda_val[i]);
      if (i%10 == 0)
         printf("\n");
   }
}


/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
	check_param();
	printf("Initializing points on the line...\n");
	init_line();

	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "Here is no cuda device\n");
		return 1;
	}

	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			printf("max grid size = %d\n", prop.maxGridSize);
			printf("max threads dim = %d\n", prop.maxThreadsDim);
			printf("max threads per block = %d\n", prop.maxThreadsPerBlock);
			maxThreadsPerBlock = prop.maxThreadsPerBlock;
		}
	}

	cudaSetDevice(i);

	/* Allocate global memory on device */
	cudaMalloc((void**)&valptr, sizeof(float) * tpoints);
	cudaMalloc((void**)&oldptr, sizeof(float) * tpoints);
	cudaMalloc((void**)&newptr, sizeof(float) * tpoints);
	cudaMalloc((void**)&datasizeptr, sizeof(int));

	cudaMemcpy(valptr, values, sizeof(float) * tpoints, cudaMemcpyHostToDevice);	
	cudaMemcpy(oldptr, oldval, sizeof(float) * tpoints, cudaMemcpyHostToDevice);
	cudaMemcpy(newptr, newval, sizeof(float) * tpoints, cudaMemcpyHostToDevice);
	cudaMemcpy(datasizeptr, &tpoints, sizeof(int), cudaMemcpyHostToDevice);

	printf("Updating all points for all time steps...\n");
	update();
	printf("Printing final results...\n");

	printfinal_cuda();

	printf("\nDone.\n\n");
	
	cudaFree(valptr);
	cudaFree(oldptr);
	cudaFree(newptr);
	
	return 0;
}
