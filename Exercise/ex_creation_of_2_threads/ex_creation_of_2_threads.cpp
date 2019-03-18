#include <iostream>
#include <cstdlib>
#include <ex_creation_of_2_threads.h>


void *CreationOf2Threads::CharPrint(void *parameters){
    CharPrintParms *p = (CharPrintParms*) parameters;
    int *count_5 = new int(0);
    *count_5 = 5*p->count;

    std::cout<<"count_5 address                = "<<&count_5<<std::endl;
    std::cout<<"count_5 address value          = "<<*&count_5<<std::endl;
    std::cout<<"count_5 pointed address        = "<<count_5<<std::endl;
    std::cout<<"count_5 pointed address value  = "<<*count_5<<std::endl;

    for(int i=0;i<p->count;++i){
        std::cout<<p->character<<std::endl;
    }
    void *addr_42 = (void *) 42;
    std::cout<<"addr_42 pointed address = "<<addr_42<<std::endl;
    std::cout<<"addr_42 address         = "<<&addr_42<<std::endl;
    std::cout<<"addr_42 address value   = "<<*(int *)&addr_42<<std::endl;

    if(p->ret_const == true){
        return addr_42;
    }else{
        return (void *) count_5;
    }
}
