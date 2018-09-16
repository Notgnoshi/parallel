#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

void print_array(const double* a, size_t len)
{
    printf("[");
    for(size_t i = 0; i < len; ++i)
    {
        printf("%f,", a[i]);
    }
    printf("]\n");
}

int32_t main()
{
    double x = 81;
    double a[9] = {0};
    double b[9] = {0};

    for(int i = 0; i < (int) sqrt(x); i++)
    {
        a[i] = 2.3 * i;
        if(i < 10)
            b[i] = a[i];
    }
    print_array(a, 9);
    print_array(b, 9);

    memset(a, 0, 9 * sizeof( double ));
    memset(b, 0, 9 * sizeof( double ));

    #pragma omp parallel for num_threads( 8 )
    for(int i = 0; i < (int) sqrt(x); i++)
    {
        a[i] = 2.3 * i;
        if(i < 10)
            b[i] = a[i];
    }
    print_array(a, 9);
    print_array(b, 9);


    int flag = 0;
    // #pragma omp parallel for num_threads( 8 )
    for(size_t i = 0; i < 9 && !flag; i++)
    {
        a[i] = 2.3 * i;
        if(a[i] < b[i])
            flag = 1;
    }

    return 0;
}
