#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char **argv) {
    int N;
    int seed;
    //int temp[N][N];
    //int** temp;
    int my_rank;  //the id of the current process
    int nprocess; //total number of processes
    double time0, time1;
    srand(seed);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    time0 = MPI_Wtime();//start timing
    //-----------Argument Parser----------//
    N = atoi(argv[1]);
    seed = atoi(argv[2]);
    //printf("N in %d of %d is %d, seed = %d\n", my_rank, nprocess, N, seed);
    time1 = MPI_Wtime();
    //printf("Argument Parser using time: %lf\n", (time1-time0));

    //-----------Initialization----------//
    time0 = MPI_Wtime();
    int temp[N][N];
    if(my_rank == 0){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[i][j] = random() >> 3; // avoid overflow
            }
        }
        printf("Before Broadcast...\n");
        printf("rank %d, N = \n", my_rank);
        for(int i=0;i<N;++i){
            for(int j=0;j<N;++j){
                printf("%2d ", temp[i][j]);
                if(j==(N-1)){
                    printf("\n");
                }
            }
        }
    }

    MPI_Bcast(temp, N*N, MPI_INT, 0, MPI_COMM_WORLD);
    time1 = MPI_Wtime();

    printf("After Broadcast...\n");
    printf("rank %d, N = \n", my_rank);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            printf("%2d ", temp[i][j]);
            if(j==(N-1)){
                printf("\n");
            }
        }
    }
    printf("Initialization using time: %lf\n", (time1-time0));
    //-----------Main Calculation----------//
    time0 = MPI_Wtime();
    int count = 0, balance = 0;
    int next[N][N];
    while (!balance) {
        count++;
        balance = 1;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int up = i - 1 < 0 ? 0 : i - 1;
                int down = i + 1 >= N ? i : i + 1;
                int left = j - 1 < 0 ? 0 : j - 1;
                int right = j + 1 >= N ? j : j + 1;
                next[i][j] = (temp[i][j] + temp[up][j] + temp[down][j] +
                    temp[i][left] + temp[i][right]) / 5;
                if (next[i][j] != temp[i][j]) {
                    balance = 0;
                }
            }
        }
        memcpy(temp, next, N * N * sizeof(int));
    }
    time1 = MPI_Wtime();

    //printf("Size: %d*%d, Seed: %d, ", N, N, seed);
    //printf("Iteration: %d, Temp: %d\n", count, temp[0][0]);
    //printf("Main Calculation using time in rank %d: %lf\n", my_rank, (time1-time0));

    MPI_Finalize();
    return 0;
}
