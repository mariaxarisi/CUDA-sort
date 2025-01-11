#ifndef VECTOR_H
#define VECTOR_H

typedef struct {
    int *arr;
    int n;
} Vector;

Vector createVec(int n);
void destroyVec(Vector v);
Vector randomVec(int n);
void printVec(Vector v);
int isSorted(Vector v);

#endif // VECTOR_H