#include "../include/vector.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Vector createVec(int n){
    Vector v;
    v.n = n;
    v.arr = (int *)malloc(n * sizeof(int));
    return v;
}

void destroyVec(Vector v){
    free(v.arr);
}

Vector randomVec(int n){

    srand(time(NULL));
    Vector v = createVec(n);
    for(int i = 0; i < n; i++){
        v.arr[i] = rand() % 1000;
    }
    return v;
}

void printVec(Vector v){
    for(int i = 0; i < v.n; i++){
        printf("%d ", v.arr[i]);
    }
    printf("\n");
}

int isSorted(Vector v){
    for(int i=0; i<v.n-1; i++){
        if(v.arr[i] > v.arr[i + 1]){
            return 0;
        }
    }
    return 1;
}