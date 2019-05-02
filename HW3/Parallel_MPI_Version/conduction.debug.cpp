#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char **argv) {
    int N;
    int seed;
    //int temp[N][N];
    //int** temp;
    int my_rank;  //the id of the current process
    int nprocess; //total number of processes
    double time0, time1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    //-----------Argument Parser----------//
    //time0 = MPI_Wtime();//start timing
    N = atoi(argv[1]);
    seed = atoi(argv[2]);
    srand(seed);
    //printf("N in %d of %d processes is %d, seed = %d\n", my_rank, nprocess, N, seed);
    //time1 = MPI_Wtime();
    //printf("Argument Parser using time: %lf\n", (time1-time0));

    //-----------Initialization----------//
    //time0 = MPI_Wtime();
    int temp[N][N];

    srand(seed);
    if(my_rank == MASTER){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                temp[i][j] = random() >> 3; // avoid overflow
            }
        }
        /*Print for debug
        for(int i=0;i<N;++i){
            for(int j=0;j<N;++j){
                printf("%d ", temp[i][j]);
                if(j==N-1){
                    printf("\n");
                }
            }
        }
        */
    }

    MPI_Bcast(temp, N*N, MPI_INT, 0, MPI_COMM_WORLD);
    //time1 = MPI_Wtime();
    //printf("Initialization using time: %lf\n", (time1-time0));
    
    //-----------Main Calculation----------//
    time0 = MPI_Wtime();
    MPI_Status status;
    int count          = 0;
    int balance        = 0;
    int avg_rows       = N/nprocess;
    int extra_rows     = N%nprocess;
    int *my_rank_start = new int [nprocess];
    int *my_rank_end   = new int [nprocess];
    int *my_rank_rows  = new int [nprocess];
    int next_tmp[(avg_rows+1)][N];
    int next[N][N]     = { {0} };
    int start_accum    = 0;

    /*
    printf("Original temp with rank %d is: \n", my_rank);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            printf("%d ", temp[i][j]);
            if(j==N-1){
                printf("\n");
            }
        }
    }
    */
    
    //work division
    for(int i=0;i<nprocess;++i){
        int i_rows = (i<(extra_rows))?(avg_rows+1):avg_rows;
        my_rank_start[i] = (i==0)?0:my_rank_end[i-1]+1;
        my_rank_end[i]   = my_rank_start[i]+(i_rows-1);
        my_rank_rows[i]  = i_rows;
    }

    while (!balance) {
        //printf("rank = %d, calculating...\n", my_rank);
        count++;
        balance = 1;

        if(count > 1){
            //printf("MPI_Bcast, rank = %d\n", my_rank);
            MPI_Bcast(temp, N*N, MPI_INT, 0, MPI_COMM_WORLD);
            //printf("MPI_Bcast done, rank = %d\n", my_rank);
        }
        
        /*
        if(my_rank == 3){
        printf("Iteration %d, rank = %d, temp = \n", count, my_rank);
        for(int i=0;i<N;++i){
            for(int j=0;j<N;++j){
                printf("%d ", temp[i][j]);
                if(j==N-1){
                    printf("\n");
                }
            }
        }
        }
        */

        for (int i = my_rank_start[my_rank]; i <= my_rank_end[my_rank]; i++) {
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

        /*
        if(my_rank == 3){
        printf("Iteration %d, rank = %d, next = \n", count, my_rank);
        for(int i=0;i<N;++i){
            for(int j=0;j<N;++j){
                printf("%d ", next[i][j]);
                if(j==N-1){
                    printf("\n");
                }
            }
        }
        }
        */

        if(my_rank == MASTER){//Master, combine all the next[][]
            int balance_from_others = 1;

            //printf("MASTER receive from othres!\n");
            for(int i=1;i<nprocess;++i){
                //printf("=======0. Receiver i = %d, next_tmp=======\n", i);
                MPI_Recv(&next_tmp[0][0], (avg_rows+1)*N, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                //printf("=======1. Receiver i = %d, next_tmp done.=======\n", i);
                int sender_rank = status.MPI_SOURCE;
                MPI_Recv(&balance_from_others, 1, MPI_INT, sender_rank, 0, MPI_COMM_WORLD, &status);
                //printf("=======2. Receiver i = %d, sender_rank = %d, balance_from_others = %d=======\n", i, sender_rank, balance_from_others);
                memcpy(&temp[my_rank_start[sender_rank]][0], next_tmp, my_rank_rows[sender_rank] * N * sizeof(int));
                //printf("=======3. Receiver i = %d, sender_rank = %d=======\n", i, sender_rank);
                //printf("=======4. Receive balance = %d, sender_rank = %d\n", balance_from_others, sender_rank);
                balance = balance*balance_from_others;
            }
            //send balance result to others
            for(int i=1;i<nprocess;++i){
                MPI_Send(&balance, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            //copy the result of Master next[][]
            memcpy(&temp[my_rank_start[MASTER]][0], next, my_rank_rows[MASTER] * N * sizeof(int));
        }else{//Workers, send the result : next[][]
            //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>0. Sender sender_rank = %d>>>>>>>>\n", my_rank);
            MPI_Send(&next[my_rank_start[my_rank]][0], my_rank_rows[my_rank]*N, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>1. Sender sender_rows= %d, sender_rank = %d>>>>>>>>\n", my_rank_rows[my_rank], my_rank);
            MPI_Send(&balance, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>2. Sender sender_balance= %d, sender_rank = %d>>>>>>>>\n", balance, my_rank);
            MPI_Recv(&balance, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
            //printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>3. Sender receive balance = %d, sender_rank = %d>>>>>>>>\n", balance, my_rank);
            //balance = 0;
        }
        //printf("> rank = %d, count = %d, balance = %d\n", my_rank, count, balance);
        //char ch = getchar();
    }//end while
    
    time1 = MPI_Wtime();
    if(my_rank == MASTER){
        printf("Size: %d*%d, Seed: %d, ", N, N, seed);
        printf("Iteration: %d, Temp: %d\n", count, temp[0][0]);
    }
    //printf("Main Calculation using time in rank %d: %lf\n", my_rank, (time1-time0));

    //printf("End while, rank = %d\n", my_rank);
    MPI_Finalize();
    return 0;
}
