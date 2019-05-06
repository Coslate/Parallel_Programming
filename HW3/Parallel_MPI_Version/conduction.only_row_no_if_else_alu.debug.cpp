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
            printf("%10d ", temp[i][j]);
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
    int balance_collect_ms  = 0;
    int balance_internal    = 0;
    int balance_from_upper  = 0;
    int balance_from_lower  = 0;
    int avg_rows            = N/nprocess;
    int extra_rows          = N%nprocess;
    int next[avg_rows+1][N] = { {0} };
    int next_process        = (my_rank+1)%nprocess;
    int prev_process        = (my_rank==0)?(nprocess-1):(my_rank-1);

    int my_rank_start[nprocess];
    int my_rank_end[nprocess];
    int my_rank_rows[nprocess];
    int upper[N+1];
    int lower[N+1];
    int padded_upper[N+1];
    int padded_lower[N+1];
    int last_rank          = 0;
    int count_work_process = 0;
    double accu_time_message_passing = 0;
    double avg_time_message_passing = 0;
    double memcpy1_time_message_passing = 0;
    double memcpy2_time_message_passing = 0;
    double balance_cal_time_message_passing = 0;
    int test_rank          = 0;

    time0 = MPI_Wtime();//start timing
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
            memcpy(padded_lower, &temp[my_rank_start[next_process]][0], N*sizeof(int));
        }else{
            memcpy(padded_lower, &temp[my_rank_end[my_rank]][0], N*sizeof(int));
        }
        memcpy(padded_upper, &temp[my_rank_start[my_rank]][0], N*sizeof(int));
    }else if(my_rank == last_rank){
        if((my_rank_rows[prev_process]>0) && (prev_process!=my_rank)){
            memcpy(padded_upper, &temp[my_rank_end[prev_process]][0], N*sizeof(int));
        }else{
            memcpy(padded_upper, &temp[my_rank_start[my_rank]][0], N*sizeof(int));
        }
        memcpy(padded_lower, &temp[my_rank_end[my_rank]][0], N*sizeof(int));
    }else{
        memcpy(padded_upper, &temp[my_rank_end[prev_process]][0], N*sizeof(int));
        memcpy(padded_lower, &temp[my_rank_start[next_process]][0], N*sizeof(int));
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
        
        if(my_rank_rows[my_rank] > 0){
            //Padding
            time0 = MPI_Wtime();//start timing

            int sub_temp[(my_rank_rows[my_rank]+2)][N];
            memcpy(sub_temp, padded_upper, N*sizeof(int)); 
            memcpy(&sub_temp[1][0], &temp[my_rank_start[my_rank]][0], N*my_rank_rows[my_rank]*sizeof(int));    
            memcpy(&sub_temp[(my_rank_rows[my_rank]+1)][0], padded_lower, N*sizeof(int));

            time1 = MPI_Wtime();//start timing
            memcpy1_time_message_passing += (time1-time0);

            /*
            if(my_rank==test_rank){
            printf("Iteration %d, rank = %d, sub_temp = \n", count, my_rank);
            for(int i=0;i<(my_rank_rows[my_rank]+2);++i){
                for(int j=0;j<N;++j){
                    printf("%10d ", sub_temp[i][j]);
                    if(j==N-1){
                        printf("\n");
                    }
                }
            }
            }
            */

            //Calculating
            time0 = MPI_Wtime();//start timing
            balance_internal = 1;
            for (int i = 1; i < (1+my_rank_rows[my_rank]); ++i) {
                for (int j = 0; j < N; j++) {
                    int up    = i - 1;
                    int down  = i + 1;
                    int left  = j - 1 < 0 ? 0 : j - 1;
                    int right = j + 1 >= N ? j : j + 1;
                    int next_i= i - 1;

                    next[next_i][j] = (sub_temp[i][j] + sub_temp[up][j] + sub_temp[down][j] + sub_temp[i][left] + sub_temp[i][right]) / 5;
                    if (next[next_i][j] != sub_temp[i][j]) {
                        balance_internal = 0;
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
                    printf("%10d ", next[i][j]);
                    if(j==N-1){
                        printf("\n");
                    }
                }
            }
            }
            */

            memcpy(lower, &next[my_rank_rows[my_rank]-1][0], N*sizeof(int));
            memcpy(upper, &next[0][0], N*sizeof(int));

            lower[N] = balance_internal;
            upper[N] = balance_internal;

            /*
            if(my_rank==test_rank){
                printf("balance_internal = %d\n", balance_internal);
                printf("lower[N] = %d\n", lower[N]);
                printf("upper[N] = %d\n", upper[N]);
                printf("lower = ");
                for(int j=0;j<N;++j){
                    printf("%d ", lower[j]);

                    if(j==N-1){
                        printf("\n");
                    }
                }
                printf("upper = ");
                for(int j=0;j<N;++j){
                    printf("%d ", upper[j]);

                    if(j==N-1){
                        printf("\n");
                    }
                }
            }
            */

            time0 = MPI_Wtime();//start timing
            if((nprocess > 1) && (count_work_process > 1)){
                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                    }else{
                        MPI_Recv(padded_lower, N+1, MPI_INT, next_process, 1, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD, &status);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Send(lower, N+1, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }else if(my_rank == last_rank){
                        MPI_Send(upper, N+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N+1, MPI_INT, prev_process, 1, MPI_COMM_WORLD);
                        MPI_Send(lower, N+1, MPI_INT, next_process, 1, MPI_COMM_WORLD);
                    }
                }

                if(my_rank % 2 == 1){//odd
                    if(my_rank == last_rank){
                        MPI_Send(upper, N+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                    }else{
                        MPI_Send(upper, N+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD);
                        MPI_Send(lower, N+1, MPI_INT, next_process, 0, MPI_COMM_WORLD);
                    }
                }else{//even
                    if(my_rank == MASTER){
                        MPI_Recv(padded_lower, N+1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                    }else if(my_rank == last_rank){
                        MPI_Recv(padded_upper, N+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }else{
                        MPI_Recv(padded_lower, N+1, MPI_INT, next_process, 0, MPI_COMM_WORLD, &status);
                        MPI_Recv(padded_upper, N+1, MPI_INT, prev_process, 0, MPI_COMM_WORLD, &status);
                    }
                }
            }else{
                balance_from_lower = 1;
                balance_from_upper = 1;
                //memcpy(padded_upper, upper, N*sizeof(int));
                //memcpy(padded_lower, lower, N*sizeof(int));
            }

            //temp & padded_lower & padded_lower update
            time1 = MPI_Wtime();//start timing
            accu_time_message_passing += (time1-time0);

            time0 = MPI_Wtime();
            if (my_rank == MASTER) {
                memcpy(padded_upper, &next[0][0], N*sizeof(int));
                padded_upper[N] = balance_internal;
            }
            if (my_rank == last_rank) {
                memcpy(padded_lower, &next[my_rank_rows[my_rank]-1][0], N*sizeof(int));
                padded_lower[N] = balance_internal;
            }
            memcpy(&temp[my_rank_start[my_rank]][0], next, my_rank_rows[my_rank]*N*sizeof(int));
            time1 = MPI_Wtime();
            memcpy2_time_message_passing += (time1-time0);

            //balnce collection
            time0 = MPI_Wtime();
            if((nprocess > 1) && (count_work_process > 1)){
                balance_from_upper = padded_upper[N];
                balance_from_lower = padded_lower[N];
                balance = balance_internal*balance_from_upper*balance_from_lower;

                /*
                if(my_rank==test_rank){
                    printf("[Internal] balance            = %d\n", balance);
                    printf("[Internal] balance_internal   = %d\n", balance_internal);
                    printf("[Internal] balance_from_lower = %d\n", balance_from_lower);
                    printf("[Internal] balance_from_upper = %d\n", balance_from_upper);
                }
                */

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
            time1 = MPI_Wtime();
            balance_cal_time_message_passing += (time1-time0);       
        }else{//end if(my_rank_rows[my_rank] > 0)
            balance = 1;
        }

        /*
        if(my_rank==test_rank){
            printf("balance            = %d\n", balance);
            printf("balance_internal   = %d\n", balance_internal);
            printf("balance_collect_ms = %d\n", balance_collect_ms);
            printf("balance_from_lower = %d\n", balance_from_lower);
            printf("balance_from_upper = %d\n", balance_from_upper);
            printf("padded_lower[N] = %d\n", padded_lower[N]);
            printf("padded_upper[N] = %d\n", padded_upper[N]);
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

        if(my_rank==test_rank){
            printf("Iteration %d, rank = %d, temp = \n", count, my_rank);
            for(int i=0;i<N;++i){
                for(int j=0;j<N;++j){
                    printf("%10d ", temp[i][j]);
                    if(j==N-1){
                        printf("\n");
                    }
                }
            }


            printf("==========================================\n");
        }
        */
        
        /*
        time0 = MPI_Wtime();
        MPI_Allreduce(&balance, &balance, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        time1 = MPI_Wtime();
        all_reduce_time_message_passing += (time1-time0);
        */
        //printf("Aft, balance = %d, my_rank = %d\n", balance, my_rank);
        //char aa = getchar();
    }//end while


    printf("avg_time_message_passing = %lf\n", avg_time_message_passing);
    printf("accu_time_message_passing = %lf\n", accu_time_message_passing);
    printf("memcpy1_time_message_passing = %lf\n", memcpy1_time_message_passing);
    printf("memcpy2_time_message_passing = %lf\n", memcpy2_time_message_passing);
    printf("balance_cal_time_message_passing = %lf\n", balance_cal_time_message_passing);
    printf("avg_time_message_passing+accu_time_message_passing+memcpy1_time_message_passing+memcpy2_time_message_passing+balance_cal_time_message_passing = %lf+%lf+%lf+%lf+%lf = %lf, rank %d\n", avg_time_message_passing, accu_time_message_passing, memcpy1_time_message_passing, memcpy2_time_message_passing, balance_cal_time_message_passing, (avg_time_message_passing+accu_time_message_passing+memcpy1_time_message_passing+memcpy2_time_message_passing+balance_cal_time_message_passing), my_rank);

    if(my_rank == MASTER){
        printf("Size: %d*%d, Seed: %d, ", N, N, seed);
        printf("Iteration: %d, Temp: %d\n", count, temp[0][0]);
    }

    //printf("End while, rank = %d\n", my_rank);
    MPI_Finalize();
    return 0;
}
