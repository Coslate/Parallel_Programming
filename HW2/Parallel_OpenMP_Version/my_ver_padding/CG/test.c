#include <stdio.h>
#include <stdlib.h>
#include <math.h>


static void conj_grad(double x[][5]);


int main(){
    double x[2][5] = {0};

    int i, j;
    for(i=0;i<2;++i){
        for(j=0;j<5;++j){
            printf("x[%d][%d] = %lf\n", i, j, x[i][j]);
        }
    }
    
    conj_grad(x);

    printf("======================================>\n");
    for(i=0;i<2;++i){
        for(j=0;j<5;++j){
            printf("x[%d][%d] = %lf\n", i, j, x[i][j]);
        }
    }
    
    return(EXIT_SUCCESS);
}


static void conj_grad(double x[][5]){
    x[1][4] = 199;
}
