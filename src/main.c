#include "../include/vector.h"
#include "../include/bitonic.h"

#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[]) {

#ifdef VERSION_V0
    printf("Running V0 implementation.\n");
#endif

#ifdef VERSION_V1
    printf("Running V1 implementation.\n");
#endif

#ifdef VERSION_V2
    printf("Running V2 implementation.\n");
#endif

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <exponent of 2>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Invalid input. Please enter a positive integer.\n");
        return 1;
    }
    
    Vector v = randomVec(1<<n);

    // printf("UNSorted vector:\n");
    // printVec(v);

    float start = clock();
    bitonicSort(v);
    float end = clock();

    // printf("Sorted vector:\n");
    // printVec(v);

    if (isSorted(v)){
        printf("The vector is sorted\n");
        printf("Execution Time: %.3f sec\n", (end - start) / CLOCKS_PER_SEC);
    } 

    destroyVec(v);

    return 0;
}