#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <ex_creation_of_2_threads.h>


int main(){
    pthread_t thread1_id;
    pthread_t thread2_id;
    CreationOf2Threads::CharPrintParms thread1_args("x", 10);
    CreationOf2Threads::CharPrintParms thread2_args("o", 20);

//    CreationOf2Threads::CharPrintParms *thread1_args = new CreationOf2Threads::CharPrintParms("x", 10);
//    CreationOf2Threads::CharPrintParms *thread2_args = new CreationOf2Threads::CharPrintParms("o", 10);

    std::cout<<"> Create a new thread to print "<<thread1_args.character<<" "<<thread1_args.count<<"times."<<std::endl;
    pthread_create(&thread1_id, NULL, &CreationOf2Threads::CharPrint, &thread1_args);

    std::cout<<"> Create a new thread to print "<<thread2_args.character<<" "<<thread2_args.count<<"times."<<std::endl;
    pthread_create(&thread2_id, NULL, &CreationOf2Threads::CharPrint, &thread2_args);

    pthread_join(thread1_id, NULL);
    std::cout<<"> Make sure the first thread has finished."<<std::endl;

    pthread_join(thread2_id, NULL);
    std::cout<<"> Make sure the second thread has finished."<<std::endl;


//    delete thread1_args;
//    delete thread2_args;
    return EXIT_SUCCESS;
}
