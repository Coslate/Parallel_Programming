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
    double time0, time1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    //-----------Argument Parser----------//
    time0 = MPI_Wtime();//start timing
    N = atoi(argv[1]);
    seed = atoi(argv[2]);
    srand(seed);
    //printf("N in %d of %d processes is %d, seed = %d\n", my_rank, nprocess, N, seed);
    time1 = MPI_Wtime();
    printf("Argument Parser using time: %lf\n", (time1-time0));

    //-----------Initialization----------//
    time0 = MPI_Wtime();
    int temp[N][N];

    srand(seed);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp[i][j] = random() >> 3; // avoid overflow
        }
    }
    
    /*
    printf("Original temp with rank %d is: \n", my_rank);
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            printf("%15d ", temp[i][j]);
            if(j==N-1){
                printf("\n");
            }
        }
    }
    */

    time1 = MPI_Wtime();
    printf("Initialization using time: %lf\n", (time1-time0));
    
    //-----------Main Calculation----------//
    MPI_Status status;
    int count               = 0;
    int balance             = 0;
    int avg_rows            = N/nprocess;
    int extra_rows          = N%nprocess;
    int next[avg_rows+1][N] = { {0} };
    int next_process        = (my_rank+1)%nprocess;
    int prev_process        = (my_rank==0)?(nprocess-1):(my_rank-1);
    int num_pad             = 1;

    int my_rank_start[nprocess];
    int my_rank_end[nprocess];
    int my_rank_rows[nprocess];
    int upper[N*num_pad];
    int lower[N*num_pad];
    int padded_upper[N*num_pad];
    int padded_lower[N*num_pad];
    int last_rank          = 0;
    int count_work_process = 0;
    double accu_time_message_passing = 0;
    double avg_time_message_passing = 0;
    double memcpy1_time_message_passing = 0;
    double memcpy2_time_message_passing = 0;
    double all_reduce_time_message_passing = 0;
    //int test_rank          = -1;

    time0 = MPI_Wtime();//start timing
    //work division
    for(int i=0;i<nprocess;++i){
        int i_rows = (i<(extra_rows))?(avg_rows+1):avg_rows;
        my_rank_start[i] = (i==0)?0:my_rank_end[i-1]+1;
        my_rank_end[i]   = my_rank_start[i]+(i_rows-1);
        my_rank_rows[i]  = i_rows;
        if(i_rows > 0){
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

    time1 = MPI_Wtime();//start timing
    printf("work division using time: %lf\n", (time1-time0));

    /*
    if(my_rank == test_rank){
        for(int i=0;i<nprocess;++i){
            printf("my_rank_start[%d] = %d\n", i, my_rank_start[i]);
            printf("my_rank_end[%d]   = %d\n", i, my_rank_end[i]);
            printf("my_rank_rows[%d]  = %d\n", i, my_rank_rows[i]);
        }
        printf("last_rank = %d\n", last_rank);
        printf("count_work_process = %d\n",count_work_process);

        printf("padded_lower = ");
        for(int j=0;j<N;++j){
            printf("%d ", padded_lower[j]);

            if(j==N-1){
                printf("\n");
            }
        }
        printf("padded_upper = ");
        for(int j=0;j<N;++j){
            printf("%d ", padded_upper[j]);

            if(j==N-1){
                printf("\n");
            }
        }
    }
    */

    while (!balance) {
        //printf("rank = %d, calculating...\n", my_rank);
        count++;
        balance = 1;
        
        if(my_rank_rows[my_rank] > 0){
            //Padding
            time0 = MPI_Wtime();//start timing

            int sub_temp[(my_rank_rows[my_rank]+(2*num_pad))][N];
            memcpy(sub_temp, padded_upper, N*num_pad*sizeof(int)); 
            memcpy(&sub_temp[num_pad][0], &temp[my_rank_start[my_rank]][0], N*my_rank_rows[my_rank]*sizeof(int));    
            memcpy(&sub_temp[(my_rank_rows[my_rank]+num_pad)][0], padded_lower, N*num_pad*sizeof(int));

            time1 = MPI_Wtime();//start timing
            memcpy1_time_message_passing += (time1-time0);

            /*
            if(my_rank==test_rank){
            printf("Iteration %d, rank = %d, sub_temp = \n", count, my_rank);
            for(int i=0;i<(my_rank_rows[my_rank]+(2*num_pad));++i){
                for(int j=0;j<N;++j){
                    printf("%15d ", sub_temp[i][j]);
                    if(j==N-1){
                        printf("\n");
                    }
                }
            }
            }
            */

            //Calculating
            time0 = MPI_Wtime();//start timing
            for (int i = num_pad; i < (num_pad+my_rank_rows[my_rank]); ++i) {
                for (int j = 0; j < N; j++) {
                    int up    = i - 1;
                    int down  = i + 1;
                    int left  = j - 1 < 0 ? 0 : j - 1;
                    int right = j + 1 >= N ? j : j + 1;
                    int next_i= i-num_pad;

                    next[next_i][j] = (sub_temp[i][j] + sub_temp[up][j] + sub_temp[down][j] + sub_temp[i][left] + sub_temp[i][right]) / 5;
                    if (next[next_i][j] != sub_temp[i][j]) {
                        balance = 0;
                    }
                }
            }
            time1 = MPI_Wtime();//start timing
            avg_time_message_passing += (time1-time0);

            /*
            if(my_rank==test_rank){
            printf("Iteration %d, rank = %d, next = \n", count, my_rank);
            for(int i=0;i<(avg_rows+1);++i){
                for(int j=0;j<N;++j){
                    printf("%15d ", next[i][j]);
                    if(j==N-1){
                        printf("\n");
                    }
                }
            }
            }
            */

            memcpy(lower, &next[my_rank_rows[my_rank]-1][0], N*num_pad*sizeof(int));
            memcpy(upper, &next[0][0], N*num_pad*sizeof(int));

            time0 = MPI_Wtime();//start timing
            if((nprocess > 1) && (count_work_process > 1)){
                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N*num_pad, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                }else{
                        MPI_Recv(padded_lower, N*num_pad, MPI_INT, next_process, 1, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N*num_pad, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Send(lower, N*num_pad, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }else if(my_rank == last_rank){
                        MPI_Send(upper, N*num_pad, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N*num_pad, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                        MPI_Send(lower, N*num_pad, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }
                }

                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Send(upper, N*num_pad, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N*num_pad, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                        MPI_Send(lower, N*num_pad, MPI_INT, next_process, 0, MPI_COMM_WORLD);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Recv(padded_lower, N*num_pad, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                    }else if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N*num_pad, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }else{
                        MPI_Recv(padded_lower, N*num_pad, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N*num_pad, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }
                }
            }else{
                //memcpy(padded_upper, upper, N*num_pad*sizeof(int));
                //memcpy(padded_lower, lower, N*num_pad*sizeof(int));
            }
            time1 = MPI_Wtime();//start timing
            accu_time_message_passing += (time1-time0);

            time0 = MPI_Wtime();
            if (my_rank == MASTER) {
                memcpy(padded_upper, &next[0][0], N*num_pad*sizeof(int));
            }
            if (my_rank == last_rank) {
                memcpy(padded_lower, &next[my_rank_rows[my_rank]-1][0], N*num_pad*sizeof(int));
            }
            memcpy(&temp[my_rank_start[my_rank]][0], next, my_rank_rows[my_rank]*N*sizeof(int));
            time1 = MPI_Wtime();
            memcpy2_time_message_passing += (time1-time0);
        }//end if(my_rank_rows[my_rank] > 0)

        //printf("Bef, balance = %d, my_rank = %d\n", balance, my_rank);
        time0 = MPI_Wtime();
        MPI_Allreduce(&balance, &balance, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        time1 = MPI_Wtime();
        all_reduce_time_message_passing += (time1-time0);
        //printf("Aft, balance = %d, my_rank = %d\n", balance, my_rank);
        //char aa = getchar();
    }//end while

    if(my_rank == MASTER){
        printf("Size: %d*%d, Seed: %d, ", N, N, seed);
        printf("Iteration: %d, Temp: %d\n", count, temp[0][0]);
    }

    /*
    printf("avg_time_message_passing = %lf\n", avg_time_message_passing);
    printf("accu_time_message_passing = %lf\n", accu_time_message_passing);
    printf("memcpy1_time_message_passing = %lf\n", memcpy1_time_message_passing);
    printf("memcpy2_time_message_passing = %lf\n", memcpy2_time_message_passing);
    printf("all_reduce_time_message_passing = %lf\n", all_reduce_time_message_passing);
    printf("avg_time_message_passing+accu_time_message_passing+memcpy1_time_message_passing+memcpy2_time_message_passing+all_reduce_time_message_passing = %lf+%lf+%lf+%lf+%lf = %lf\n", avg_time_message_passing, accu_time_message_passing, memcpy1_time_message_passing, memcpy2_time_message_passing, all_reduce_time_message_passing, (avg_time_message_passing+accu_time_message_passing+memcpy1_time_message_passing+memcpy2_time_message_passing+all_reduce_time_message_passing));
    */

    //printf("End while, rank = %d\n", my_rank);
    MPI_Finalize();
    return 0;
}
