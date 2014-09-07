#include <string.h>
#include <math.h>
#include "benchmark.h"
#include <omp.h>

void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    /* TODO: write a faster version of eig */
    



//Power iteration is a standard way to find the dominant eigenvalue of a matrix.
//Pick a random vector v, then hit it with P repeatedly until you stop seeing
//it change very much. You want to periodically divide v by sqrt(v^T v) to normalise it.


        for (size_t k = 0; k < iters; k += 1) {
        /* v_k = Au_{k-1} */
        memset(v, 0, n * n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                for (size_t j = 0; j < n; j += 1) {
                    v[i + l*n] += u[j + l*n] * A[i + n*j];
                }
            }
        }
        /* mu_k = ||v_k|| */
        float mu[n];
        memset(mu, 0, n * sizeof(float));

    double global_mu_k = 0.0;
    #pragma omp parallel
	{
		#pragma omp for reduction(+:global_mu_k) 
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                global_mu_k += v[i + l*n] * v[i + l*n];
            }
            mu[l] = sqrt(global_mu_k);
            global_mu_k=0;
        }
    }



        /* u_k = v_k / mu_k */

	// double global_u_k = 0.0;
    #pragma omp parallel
	{
		#pragma omp for   
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                u[i + l*n] = v[i + l*n] / mu[l];
            }
        }
    }
}
}
