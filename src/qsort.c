#include "../include/vector.h"

#include <stdlib.h>

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

void bitonicSort(Vector v) {
    qsort(v.arr, v.n, sizeof(int), compare);
}