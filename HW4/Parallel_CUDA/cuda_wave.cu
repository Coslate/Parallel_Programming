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
#define PI  3.14159265
#define FAC 6.28318530

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

int threads_per_block;
float *values_d;

/**********************************************************************
 *	Handle errors function
 *********************************************************************/
static void HandleError( cudaError_t err,
                         const char *file,
                         const int line ) {
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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
    /* Update values for each time step */
    for (i = 1; i<= nsteps; i++) {
        /* Update points along line for this time step */
        for (j = 1; j <= tpoints; j++) {
            /* global endpoints */
            if ((j == 1) || (j  == tpoints))
                newval[j] = 0.0;
            else
                do_math(j);
        }

        /* Update old values with new values */
        for (j = 1; j <= tpoints; j++) {
            oldval[j] = values[j];
            values[j] = newval[j];
        }
    }
}

__global__ void cuda_update(float *values_d, const int tpoints, const int nsteps){
    int thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    float new_val;
    float curr_val;
    float old_val;

    if(thread_id < tpoints){
        //init_line()
        float x           = (float)thread_id/(tpoints - 1);
        int residue_nstep = nsteps%128;
        curr_val          = __sinf(FAC * x);
        old_val           = curr_val;
        
        //update()
        /*
        for(int i = 0; i < nsteps; ++i){
            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;
        }
        */
        //unroll 128
        for(int i = 0; i < residue_nstep; ++i){
            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;
        }
        for(int i = residue_nstep; i < nsteps; i+=128){
            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;
            
            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;
            
            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;

            new_val = 1.82*curr_val-old_val;
            old_val = curr_val;
            curr_val= new_val;           
        }

        values_d[thread_id] = curr_val;
    }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal()
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}

void printfinal_cuda(const int tpoints, float* values){
   int i;

   for (i = 0; i < tpoints; i++) {
      printf("%6.4f ", values[i]);
      if ((i+1)%10 == 0)
         printf("\n");
   }
}

// Print device properties
void PrintDevProp(cudaDeviceProp &devProp){
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block: %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:  %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %zu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("Memory Clock Rate (KHz):       %d\n", devProp.memoryClockRate);
    printf("Memory Bus Width (bits):       %d\n", devProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):  %f\n\n", 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);
    return;
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[])
{
    sscanf(argv[1],"%d",&tpoints);
    sscanf(argv[2],"%d",&nsteps);
    check_param();

    // Number of CUDA devices
    int devCount;
    int blocks_num;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&devCount);
    if(devCount==0){
        fprintf(stderr, "%s", "Error: No NVIDIA GPU device found!\n");
        return EXIT_FAILURE;
    }
 
    /*
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        if(cudaGetDeviceProperties(&devProp, i) == cudaSuccess){
            PrintDevProp(devProp);
        }
    }
    */

    //Set CUDA related parameters
    if(cudaGetDeviceProperties(&devProp, 0) == cudaSuccess){
        threads_per_block = devProp.maxThreadsPerBlock;
    }else{
        fprintf(stderr, "%s", "Error: Get GPU Property, maxThreadsPerBlock, fails.\n");
        return EXIT_FAILURE;
    }

    cudaSetDevice(0);
    blocks_num = (tpoints+threads_per_block-1)/threads_per_block;

    //Malloc mem in GPU for values_d
    HANDLE_ERROR(cudaMalloc((void**) &values_d, tpoints*sizeof(float)));

    //Initialization - already merged into kernel function
    printf("Initializing points on the line...\n");
    //init_line();

    //Update - kernel function
    printf("Updating all points for all time steps...\n");
    cuda_update<<<blocks_num, threads_per_block>>>(values_d, tpoints, nsteps);
    //Copy back from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(values, values_d, tpoints*sizeof(float), cudaMemcpyDeviceToHost));

    //Print final results
    printf("Printing final results...\n");
    printfinal_cuda(tpoints, values);
    printf("\nDone.\n\n");
	
    return EXIT_SUCCESS;
}
