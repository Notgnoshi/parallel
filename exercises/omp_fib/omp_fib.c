/* File:
 *    omp_fib.c
 *
 * Purpose:
 *    Illustrate basic use of OpenMP:
 *    Naive solution to fibonacci number
 *
 * Input:
 *    none
 * Output:
 *    array of fibonacci numbers.
 *
 * Compile:  gcc -g -Wall -fopenmp -o fib omp_fib.c
 * Usage:    ./fib
 *
 * Notes:
 *   1.  There is a serious bug
 *
 * IPP:   Section 5.5.2 (p. 227 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int n = 10;

int main(int argc, char *argv[]){
	int thread_count = omp_get_num_procs();
	int i;
	int fib[n];

	/* Initial condition */
	fib[0] = fib[1] = 1;
	for(i=2; i<n; i++)
		fib[i] = 0;

#   pragma omp parallel for num_threads(thread_count)
	for (i = 2; i<n; i++)
	{
		// Calculating the fibonacci sequence this way is inherently a sequential problem.
		fib[i] = fib[i-1] + fib[i-2];
	}

	for(i=0; i<n; i++)
		printf("%d ", fib[i]);
	printf("\n");

	return 0;
}
