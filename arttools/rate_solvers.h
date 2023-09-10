#ifndef MATH
#define MATh
#include <math.h>
#endif

double brent (double rate, double exposure, double *sp, double *bp, int ssize);

double get_phc_solution(double r, double e, double *s, double *b, int size);

double get_phc_solution_pkr(double r, double e, double *pk, int size);

double get_lkl_pkr(double r, double *pk, int size);

typedef double (*fptr)(double, double, double *, double *, int);
