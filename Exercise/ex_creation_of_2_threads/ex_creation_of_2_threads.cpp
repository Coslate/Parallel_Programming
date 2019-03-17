#include <iostream>
#include <cstdlib>
#include <ex_creation_of_2_threads.h>


void *CreationOf2Threads::CharPrint(void *parameters){
    CharPrintParms *p = (CharPrintParms*) parameters;

    for(int i=0;i<p->count;++i){
        std::cout<<p->character<<std::endl;
    }

    return NULL;
}
