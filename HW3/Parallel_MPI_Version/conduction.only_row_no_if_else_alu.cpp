#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MASTER 0

int main(int argc, char **argv) {
    int N;
    int seed;
    int my_rank;  //the id of the current process
    int nprocess; //total number of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    //-----------Argument Parser----------//
    N = atoi(argv[1]);
    seed = atoi(argv[2]);
    srand(seed);

    //-----------Initialization----------//
    int temp[N][N];
    srand(seed);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp[i][j] = random() >> 3; // avoid overflow
        }
    }

    //-----------Main Calculation----------//
    MPI_Status status;
    int count               = 0;
    int balance             = 0;
    int balance_collect_ms  = 0;
    int balance_internal    = 0;
    int balance_from_upper  = 0;
    int balance_from_lower  = 0;
    int avg_rows            = N/nprocess;
    int extra_rows          = N%nprocess;
    int next[avg_rows+1][N] = { {0} };
    int next_process        = (my_rank+1)%nprocess;
    int prev_process        = (my_rank==0)?(nprocess-1):(my_rank-1);
    int num_pad             = 1;

    int my_rank_start[nprocess];
    int my_rank_end[nprocess];
    int my_rank_rows[nprocess];
    int upper[N*num_pad+1];
    int lower[N*num_pad+1];
    int padded_upper[N*num_pad+1];
    int padded_lower[N*num_pad+1];
    int last_rank          = 0;
    int count_work_process = 0;

    //work division
    for(int i=0;i<nprocess;++i){
        my_rank_rows[i]  = (i<(extra_rows))?(avg_rows+1):avg_rows;
        my_rank_start[i] = (i==0)?0:my_rank_end[i-1]+1;
        my_rank_end[i]   = my_rank_start[i]+(my_rank_rows[i]-1);
        if(my_rank_rows[i] > 0){
            last_rank = i;
            ++count_work_process;
        }
    }

    //padded initialization
    if(my_rank == MASTER){
        if((my_rank_rows[next_process]>0) && (next_process!=my_rank)){
            memcpy(padded_lower, &temp[my_rank_start[next_process]][0], N*num_pad*sizeof(int));
        }else{
            memcpy(padded_lower, &temp[my_rank_end[my_rank]][0], N*num_pad*sizeof(int));
        }
        memcpy(padded_upper, &temp[my_rank_start[my_rank]][0], N*num_pad*sizeof(int));
    }else if(my_rank == last_rank){
        if((my_rank_rows[prev_process]>0) && (prev_process!=my_rank)){
            memcpy(padded_upper, &temp[my_rank_end[prev_process]][0], N*num_pad*sizeof(int));
        }else{
            memcpy(padded_upper, &temp[my_rank_start[my_rank]][0], N*num_pad*sizeof(int));
        }
        memcpy(padded_lower, &temp[my_rank_end[my_rank]][0], N*num_pad*sizeof(int));
    }else{
        memcpy(padded_upper, &temp[my_rank_end[prev_process]][0], N*num_pad*sizeof(int));
        memcpy(padded_lower, &temp[my_rank_start[next_process]][0], N*num_pad*sizeof(int));
    }

    while (!balance) {
        count++;
        
        if(my_rank_rows[my_rank] > 0){
            //Padding
            int sub_temp[(my_rank_rows[my_rank]+(2*num_pad))][N];
            memcpy(sub_temp, padded_upper, N*num_pad*sizeof(int)); 
            memcpy(&sub_temp[num_pad][0], &temp[my_rank_start[my_rank]][0], N*my_rank_rows[my_rank]*sizeof(int));    
            memcpy(&sub_temp[(my_rank_rows[my_rank]+num_pad)][0], padded_lower, N*num_pad*sizeof(int));

            //Calculating
            balance_internal = 1;
            for (int i = num_pad; i < (num_pad+my_rank_rows[my_rank]); ++i) {
                for (int j = 0; j < N; j++) {
                    int up    = i - 1;
                    int down  = i + 1;
                    int left  = j - 1 < 0 ? 0 : j - 1;
                    int right = j + 1 >= N ? j : j + 1;
                    int next_i= i-num_pad;

                    next[next_i][j] = (sub_temp[i][j] + sub_temp[up][j] + sub_temp[down][j] + sub_temp[i][left] + sub_temp[i][right]) / 5;
                    if (next[next_i][j] != sub_temp[i][j]) {
                        balance_internal = 0;
                    }
                }
            }

            memcpy(lower, &next[my_rank_rows[my_rank]-1][0], N*num_pad*sizeof(int));
            memcpy(upper, &next[0][0], N*num_pad*sizeof(int));

            lower[N*num_pad] = balance_internal;
            upper[N*num_pad] = balance_internal;

            if((nprocess > 1) && (count_work_process > 1)){
                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N*num_pad+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                    }else{
                        MPI_Recv(padded_lower, N*num_pad+1, MPI_INT, next_process, 1, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N*num_pad+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Send(lower, N*num_pad+1, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }else if(my_rank == last_rank){
                        MPI_Send(upper, N*num_pad+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N*num_pad+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                        MPI_Send(lower, N*num_pad+1, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }
                }

                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Send(upper, N*num_pad+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N*num_pad+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                        MPI_Send(lower, N*num_pad+1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Recv(padded_lower, N*num_pad+1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                    }else if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N*num_pad+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }else{
                        MPI_Recv(padded_lower, N*num_pad+1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N*num_pad+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }
                }
            }else{
                balance_from_lower = 1;
                balance_from_upper = 1;
                //memcpy(padded_upper, upper, N*num_pad*sizeof(int));
                //memcpy(padded_lower, lower, N*num_pad*sizeof(int));
            }

            //temp & padded_lower & padded_lower update
            if (my_rank == MASTER) {
                memcpy(padded_upper, &next[0][0], N*num_pad*sizeof(int));
                padded_upper[N*num_pad] = balance_internal;
            }
            if (my_rank == last_rank) {
                memcpy(padded_lower, &next[my_rank_rows[my_rank]-1][0], N*num_pad*sizeof(int));
                padded_lower[N*num_pad] = balance_internal;
            }
            memcpy(&temp[my_rank_start[my_rank]][0], next, my_rank_rows[my_rank]*N*sizeof(int));

            //balnce collection
            if((nprocess > 1) && (count_work_process > 1)){
                balance_from_upper = padded_upper[N*num_pad];
                balance_from_lower = padded_lower[N*num_pad];
                balance = balance_internal*balance_from_upper*balance_from_lower;

                if(my_rank == MASTER){
                    for(int i=1;i<=last_rank;i+=2){
                        MPI_Recv(&balance_collect_ms, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
                        balance = balance*balance_collect_ms;
                    }
                    for(int i=1;i<=last_rank;++i){
                        MPI_Send(&balance           , 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                    }
                }else if(my_rank % 2 == 1){
                    MPI_Send(&balance           , 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD);
                    MPI_Recv(&balance_collect_ms, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &status);
                    balance = balance_collect_ms;
                }else{
                    MPI_Recv(&balance_collect_ms, 1, MPI_INT, MASTER, 2, MPI_COMM_WORLD, &status);
                    balance = balance_collect_ms;
                }
            }else{//only one process 
                balance = balance_internal;
            }
        }else{//end if(my_rank_rows[my_rank] > 0)
            balance = 1;
        }
        
    }//end while


    if(my_rank == MASTER){
        printf("Size: %d*%d, Seed: %d, ", N, N, seed);
        printf("Iteration: %d, Temp: %d\n", count, temp[0][0]);
    }

    MPI_Finalize();
    return 0;
}
