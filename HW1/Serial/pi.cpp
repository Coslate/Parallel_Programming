#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <random>
#include <map>
#include <iomanip>
#include <ctime>

//Shared Variables
long long int number_in_circles     = 0;
long long int number_of_tosses      = 0;
long long int each_thread_work_load = 0;
int thread_num                      = 0;

const int range_from  = -1;
const int range_to    = 1;


void PrintHelp(){
    std::cerr<<"Error: This pi.o takes two input arguments."<<std::endl;
    std::cerr<<"       Please use as the format: ./pi.o [arg1] [arg1]"<<std::endl;
    std::cerr<<"       [arg1]: The number of cores that you want to run this program on."<<std::endl;
    std::cerr<<"       [arg2]: The number of tosses that you want to make for the estimation."<<std::endl;
}



void *ThreadFunction(void *rank){
    long int thread_id = (long int) rank;
    long long int my_start = thread_id*each_thread_work_load;
    long long int my_end   = (thread_id+1)*each_thread_work_load;

//    std::random_device                    rand_dev;
//    std::mt19937                          generator(rand_dev());
//    std::uniform_real_distribution<>      distr(range_from, range_to);

    for(long long int i=my_start;i<my_end;++i){
        float x = range_from + (float)(rand())/(float)(RAND_MAX)*(float)(range_to-range_from);
        float y = range_from + (float)(rand())/(float)(RAND_MAX)*(float)(range_to-range_from);
//        float x = range_from + static_cast<float>(rand())/static_cast<float>(RAND_MAX)*static_cast<float>(range_to-range_from);
//        float y = range_from + static_cast<float>(rand())/static_cast<float>(RAND_MAX)*static_cast<float>(range_to-range_from);
//        float x = (float)distr(generator);
//        float y = (float)distr(generator);
        float distance = sqrt(x*x+y*y);
//        float distance = 0;        

        if(distance <= 1){
            ++number_in_circles;
        }
    }

    return NULL;
}

int main(int argc, char *argv[]){
    if(argc != 3){
        PrintHelp();
        return EXIT_FAILURE;
    }

    float pi_esimate     = 0;
    unsigned seed         = (unsigned)time(NULL);//get time sequence
    srand(seed);//use current time as seed

//    thread_num            = std::stoi(argv[1]);
    thread_num            = 1;
    number_of_tosses      = std::stoll(argv[2]);
    each_thread_work_load = number_of_tosses/thread_num;

    ThreadFunction((void *)thread_num);
    pi_esimate = 4*((float)number_in_circles)/(float)(number_of_tosses);

    std::cout.precision(7);
    std::cout<<pi_esimate<<std::fixed<<std::endl;
    return EXIT_SUCCESS;
}
