#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <map>
#include <iomanip>
#include <ctime>
#include <pthread.h>
#include <stdint.h>

#define RAND_MAX_HALF_PRIV (RAND_MAX/2.f)

//Shared Variables
long long int number_in_circles     = 0;
long long int number_of_tosses      = 0;
long  int     thread_num            = 0;
pthread_mutex_t mutex;
long long int each_thread_work_load = 0;

/*
inline void PrintHelp(){
    std::cerr<<"Error: This pi.o takes two input arguments."<<std::endl;
    std::cerr<<"       Please use as the format: ./pi.o [arg1] [arg1]"<<std::endl;
    std::cerr<<"       [arg1]: The number of cores that you want to run this program on."<<std::endl;
    std::cerr<<"       [arg2]: The number of tosses that you want to make for the estimation."<<std::endl;
}
*/

void *ThreadFunction(void *rank){
    long int thread_id          = (long int) rank;
    long long int my_local_sum  = 0;
    unsigned int thread_seed    = (unsigned int) thread_id;

    for(long long int i=0;i<each_thread_work_load;++i){
        float x = -1.f + (float)(rand_r(&thread_seed))/(float)(RAND_MAX_HALF_PRIV);
        float y = -1.f + (float)(rand_r(&thread_seed))/(float)(RAND_MAX_HALF_PRIV);
        float distance = x*x+y*y;

        if(distance <= 1.f){
            ++my_local_sum;
        }
    }

    pthread_mutex_lock(&mutex);
    number_in_circles += my_local_sum;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char *argv[]){
/*    if(argc != 3){
        PrintHelp();
        return EXIT_FAILURE;
    }*/

    float pi_esimate          = 0;
    long int create_thread_num= 0;
    //long int seed_t       = time(NULL);

    thread_num                = std::stoi(argv[1]);
    number_of_tosses          = std::stoll(argv[2]);
    each_thread_work_load     = number_of_tosses/thread_num;
    create_thread_num         = thread_num-1;
    pthread_t *thread_handles = new pthread_t [thread_num-1];
    pthread_mutex_init(&mutex, NULL);

    for(long int thread_id=0;thread_id<create_thread_num;++thread_id){
        pthread_create(&thread_handles[thread_id], NULL, ThreadFunction, (void *)(thread_id));
    }

    ThreadFunction((void *)(thread_num));

    for(long int thread_id=0;thread_id<create_thread_num;++thread_id){
        pthread_join(thread_handles[thread_id], NULL);
    }

    pi_esimate = 4.f*((float)number_in_circles)/(float)(number_of_tosses);

    printf("%.6f\n", pi_esimate);
    delete thread_handles;
    pthread_mutex_destroy(&mutex);
    return EXIT_SUCCESS;
}
