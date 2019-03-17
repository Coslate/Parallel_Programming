#ifndef _EX_CREATION_OF_2_THREADS_H_
#define _EX_CREATION_OF_2_THREADS_H_

namespace CreationOf2Threads{
    struct CharPrintParms {
        std::string character;
        int         count;
        bool        ret_const;

        CharPrintParms(std::string a, int b, bool c):character(a), count(b), ret_const(c){};
        CharPrintParms():character(""), count(0), ret_const(true){};
    };

    void *CharPrint(void *parameters=0);
}

#endif
