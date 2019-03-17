#ifndef _EX_CREATION_OF_2_THREADS_H_
#define _EX_CREATION_OF_2_THREADS_H_

namespace CreationOf2Threads{
    struct CharPrintParms {
        std::string character;
        int         count;

        CharPrintParms(std::string a, int b):character(a), count(b){};
        CharPrintParms():character(""), count(0){};
    };

    void *CharPrint(void *parameters=0);
}

#endif
