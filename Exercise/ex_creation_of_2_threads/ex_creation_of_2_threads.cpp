#include <iostream>
#include <cstdlib>
#include <ex_creation_of_2_threads.h>


void *CreationOf2Threads::CharPrint(void *parameters){
    CharPrintParms *p = (CharPrintParms*) parameters;
    int *count_5 = new int(0);
    *count_5 = 5*p->count;

    std::cout<<"count_5 addr = "<<count_5<<std::endl;
    std::cout<<"count_5 val  = "<<*count_5<<std::endl;

    for(int i=0;i<p->count;++i){
        std::cout<<p->character<<std::endl;
    }

    if(p->ret_const == true){
        return (void *) 42;
    }else{
        return (void *) count_5;
    }
}
