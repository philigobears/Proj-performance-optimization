#include <string.h>
#include <math.h>

#include <omp.h>
#include <emmintrin.h>
#include "benchmark.h"
#include "pmmintrin.h"


#include <stdio.h>

// try padding
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {

    
    //omp_set_dynamic(0);     
    //omp_set_num_threads(16);
    register __m128 aTmp, vTmp, uTmp, muTmp;
    //dynamic padding
    int N;
    
    int index = n%64;

    if (n <= 4) {
        N = 4;
    } else if (n <= 16){
        N = 16;

    } else if (n <= 32) {
        N = 32;
    } else if (n < 64) {
        N = 64;
    }else if (index == 0 || index == 4 || index == 16 || index == 48) {
        N = n;
    } else if (index < 4) {
        N = (n/4 + 1)*4;
    } else if (index < 16) {
        N = (n/16 + 1)*16;
    } else if (index < 32) {
        N = (n/32 + 1)*32;
    } else if (index < 48) {
        N = (n/48 + 1)*48;
    } else {
        N = (n/64 + 1)*64;
    }
    

    // padding the data
    float *v_p = (float*) malloc(N*N*sizeof(float));
    float *A_p = (float*) malloc(N*N*sizeof(float));
    //float *u_p = (float*) malloc(N*N*sizeof(float));

    memset(v_p, 0, sizeof(float) * N * N);
    memset(A_p, 0, sizeof(float) * N * N);
    //memset(u_p, 0, sizeof(float) * N * N);

    #pragma omp parallel
    {
        #pragma omp for shared(v_p, v, A, A_p)

        for (int i = 0; i < n; i++)
        {

            memcpy((v_p+i*N), (v+i*n), n*sizeof(float));
            memcpy((A_p+i*N), (A+i*n), n*sizeof(float));
            
        }
    }

    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v_p, 0, N * N * sizeof(float));
        #pragma omp parallel for shared(A_p, v_p, u) private(aTmp, vTmp, uTmp)
        
        for (size_t l = 0; l < n; l += 1) {

            for (size_t j = 0; j < n; j += 1) {
                int Nj = N*j;
                //int ln = l*n;
                int lN = l*N;
                uTmp = _mm_load1_ps(u + j+l*n);
                
                for (size_t i = 0; i < N/64*64; i += 64) {
                    float *add1 = A_p + Nj + i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);

                    aTmp = _mm_loadu_ps(add1 +32);
                    vTmp = _mm_loadu_ps(add2 +32);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +32),vTmp);

                    aTmp = _mm_loadu_ps(add1 +36);
                    vTmp = _mm_loadu_ps(add2 +36);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +36),vTmp);

                    aTmp = _mm_loadu_ps(add1 +40);
                    vTmp = _mm_loadu_ps(add2 +40);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +40),vTmp);

                    aTmp = _mm_loadu_ps(add1 +44);
                    vTmp = _mm_loadu_ps(add2 +44);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +44),vTmp);

                    aTmp = _mm_loadu_ps(add1 +48);
                    vTmp = _mm_loadu_ps(add2 +48);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +48),vTmp);

                    aTmp = _mm_loadu_ps(add1 +52);
                    vTmp = _mm_loadu_ps(add2 +52);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +52),vTmp);

                    aTmp = _mm_loadu_ps(add1 +56);
                    vTmp = _mm_loadu_ps(add2 +56);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +56),vTmp);

                    aTmp = _mm_loadu_ps(add1 +60);
                    vTmp = _mm_loadu_ps(add2 +60);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +60),vTmp);
                    
                }
                // edge case
                for (size_t i = N/64*64; i < N/32*32; i += 32) {
                    float *add1 = A_p + Nj+i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);
                }

                for (size_t i = N/32*32; i < N/16*16; i += 16) {
                    float *add1 = A_p + Nj+i;
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);
                }

                for (size_t i = N/16*16; i < N; i += 4) {
                    float *add2 = v_p + i + lN;
                    aTmp = _mm_loadu_ps(A_p + Nj+i);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);
                }
                
            }
        }
        
        float mu[n];
        float tmp[4];
        memset(mu, 0, n * sizeof(float));
        
        #pragma omp parallel for shared(v_p, mu) private(vTmp, uTmp,tmp)
        
        for (size_t l = 0; l < n; l += 1) {
            //__m128  vTmp, uTmp;
            
            memset(tmp, 0, 4 * sizeof(float));
            uTmp = _mm_setzero_ps();
            int lN = l*N;
            for (size_t i = 0; i < N/64*64; i += 64) {
                float *add1 = v_p + i + lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/64*64; i < N/32*32; i += 32) {
                float *add1 = v_p + i +lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/32*32; i < N/16*16; i += 16) {
                float *add1 = v_p + i +lN;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = N/16*16; i < N; i += 4) {
                
                vTmp = _mm_loadu_ps(v_p + i +lN);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }
            
            _mm_storeu_ps(tmp, uTmp);
            mu[l] = tmp[0] +tmp[1] + tmp[2] +tmp[3];
            
            //mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for shared(mu)
        
        for (size_t l = 0; l < n/4*4; l += 4) {
            muTmp = _mm_loadu_ps(mu + l);
            _mm_storeu_ps(mu + l, _mm_sqrt_ps(muTmp));
        }
        for (size_t l = n/4*4; l < n; l ++) {
            mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for shared(v_p, u) private(vTmp, uTmp, muTmp)
        for (size_t l = 0; l < n; l += 1) {
            muTmp = _mm_set_ps(mu[l] , mu[l], mu[l], mu[l]);
            int ln = l*n;
            int lN = l*N;
            for (size_t i = 0; i < n/64*64; i += 64) {
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                _mm_storeu_ps(add2 +32, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                _mm_storeu_ps(add2 +36, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                _mm_storeu_ps(add2 +40, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                _mm_storeu_ps(add2 +44, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                _mm_storeu_ps(add2 +48, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                _mm_storeu_ps(add2 +52, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                _mm_storeu_ps(add2 +56, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                _mm_storeu_ps(add2 +60, _mm_div_ps(vTmp, muTmp));

            }

            for (size_t i = n/64*64; i < n/32*32; i += 32) {
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/32*32; i < n/16*16; i +=16){
                float *add1 = v_p +i +lN;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/16*16; i < n/4*4; i +=4) {
                vTmp = _mm_loadu_ps( v_p + i + lN);
                _mm_storeu_ps( u + i + ln, _mm_div_ps(vTmp, muTmp));
            }
            for (size_t i = n/4*4; i < n; i++) {
                u[i + ln] = v_p[i + lN] / mu[l];
            }
        }
    }
    // convert back to the original one
    #pragma omp parallel
    {
        #pragma omp for shared(v, v_p)

        for (int i = 0; i < n; i++)
        {
            memcpy((v+i*n), (v_p+i*N), n*sizeof(float));
            
        }
    }
    free(v_p);

    free(A_p);
}

/**/
/**
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {

    // do the transpose

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(16);
    __m128 aTmp, vTmp, uTmp, muTmp, tmpMatrx;
    float tmp[4];
    float *A_p = (float*) malloc(n*n*sizeof(float));
    #pragma omp parallel
    {
        #pragma omp for private(i, j, A, A_p, n)

        for (int i = 0; i < n; i++)
        {
            // then we need to tranpose the A
            for (int j = 0; j < n; j++){
                A_p[j + i*n] = A[i + j*n];
            }
        }
    }
    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v, 0, n * n * sizeof(float));
        #pragma omp parallel for private(aTmp, uTmp, tmpMatrx, tmp)
        
        for (size_t j = 0; j < n; j += 1) {
            
            for (size_t i = 0; i < n; i += 1) {
                
                memset(tmp, 0, 4 * sizeof(float));
                tmpMatrx = _mm_setzero_ps();
                int in = i*n;
                int jn = j*n;
                for (size_t l = 0; l < n/16*16; l += 16) {
                    float *add1 = in + A_p + l;
                    float *add2 = u + l +jn;

                    aTmp = _mm_loadu_ps(add1);
                    uTmp = _mm_loadu_ps(add2);
                    tmpMatrx = _mm_add_ps(tmpMatrx, _mm_mul_ps(uTmp, aTmp));

                    aTmp = _mm_loadu_ps(add1 + 4);
                    uTmp = _mm_loadu_ps(add2 + 4);
                    tmpMatrx = _mm_add_ps(tmpMatrx, _mm_mul_ps(uTmp, aTmp));

                    aTmp = _mm_loadu_ps(add1 + 8);
                    uTmp = _mm_loadu_ps(add2 + 8);
                    tmpMatrx = _mm_add_ps(tmpMatrx, _mm_mul_ps(uTmp, aTmp));

                    aTmp = _mm_loadu_ps(add1 + 12);
                    uTmp = _mm_loadu_ps(add2 + 12);
                    tmpMatrx = _mm_add_ps(tmpMatrx, _mm_mul_ps(uTmp, aTmp));     
                }
                float *tmpAdd1 = A_p + in;
                float *tmpAdd2 = u + jn;
                for (size_t l = (n/16)*16; l < n/4*4; l += 4) {
                    
                    aTmp = _mm_loadu_ps(tmpAdd1 + l);
                    uTmp = _mm_loadu_ps(tmpAdd2 + l);
                    tmpMatrx = _mm_add_ps(tmpMatrx, _mm_mul_ps(uTmp, aTmp));
                }
                _mm_storeu_ps(tmp, tmpMatrx);
                *(v + i+j*n) = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                for (size_t l = n/4*4; l < n; l++) {
                    v[i + j*n] += u[l+jn] * A_p[l + in];
                }
            }
        }
        
        float mu[n];
        
        memset(mu, 0, n * sizeof(float));
        #pragma omp parallel for private(vTmp, uTmp,tmp)
        for (size_t l = 0; l < n; l += 1) {
            
            memset(tmp, 0, 4 * sizeof(float));
            uTmp = _mm_setzero_ps();
            int ln = l*n;
            for (size_t i = 0; i < n/16*16; i += 16) {
                float *add1 = v + i + ln;

                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = n/16*16; i < n/4*4; i+=4) {
                vTmp = _mm_loadu_ps(v + i+ln);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }
            _mm_storeu_ps(tmp, uTmp);
            mu[l] = tmp[0] +tmp[1] + tmp[2] +tmp[3];
            // broundary case
            for (size_t i = n/4*4; i < n; i++) {
                mu[l] += v[i + ln] * v[i + ln];
            }
            mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for private(vTmp, muTmp)
        for (size_t l = 0; l < n; l += 1) {
            muTmp = _mm_set_ps(mu[l] , mu[l], mu[l], mu[l]);
            int ln = l*n;
            for (size_t i = 0; i < n/16*16; i += 16) {
                float *add1 = v + i + ln;
                float *add2 = u + i + ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));
            }
            for (size_t i = n/16*16; i < n/4*4; i +=4) {
                vTmp = _mm_loadu_ps(v + i + ln);
                _mm_storeu_ps(u + i + ln, _mm_div_ps(vTmp, muTmp));
            }
            for (size_t i = n/4*4; i < n; i++) {
                u[i + l*n] = v[i + ln] / mu[l];
            }
        }
    }
}

/**

// BEST EDITION
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {

    
    //omp_set_dynamic(0);     // Explicitly disable dynamic teams
    //omp_set_num_threads(16);
    register __m128 aTmp, vTmp, uTmp, muTmp;

    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v, 0, n * n * sizeof(float));
        #pragma omp parallel for shared(A, v, u) private(aTmp, vTmp, uTmp)
        
        for (size_t l = 0; l < n; l += 1) {

            for (size_t j = 0; j < n; j += 1) {
                int nj = n*j;
                int ln = l*n;
                uTmp = _mm_load1_ps(u + j+ln);
                float *add_half1 = A + nj;
                float *add_half2 = v + ln;
                for (size_t i = 0; i < n/64*64; i += 64) {
                    float *add1 = add_half1 + i;
                    float *add2 = add_half2 + i;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);

                    aTmp = _mm_loadu_ps(add1 +32);
                    vTmp = _mm_loadu_ps(add2 +32);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +32),vTmp);

                    aTmp = _mm_loadu_ps(add1 +36);
                    vTmp = _mm_loadu_ps(add2 +36);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +36),vTmp);

                    aTmp = _mm_loadu_ps(add1 +40);
                    vTmp = _mm_loadu_ps(add2 +40);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +40),vTmp);

                    aTmp = _mm_loadu_ps(add1 +44);
                    vTmp = _mm_loadu_ps(add2 +44);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +44),vTmp);

                    aTmp = _mm_loadu_ps(add1 +48);
                    vTmp = _mm_loadu_ps(add2 +48);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +48),vTmp);

                    aTmp = _mm_loadu_ps(add1 +52);
                    vTmp = _mm_loadu_ps(add2 +52);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +52),vTmp);

                    aTmp = _mm_loadu_ps(add1 +56);
                    vTmp = _mm_loadu_ps(add2 +56);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +56),vTmp);

                    aTmp = _mm_loadu_ps(add1 +60);
                    vTmp = _mm_loadu_ps(add2 +60);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +60),vTmp);
                    
                }
                // edge case
                for (size_t i = n/64*64; i < n/32*32; i += 32) {
                    float *add1 = add_half1 + i;
                    float *add2 = add_half2 + i;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);

                    aTmp = _mm_loadu_ps(add1 +16);
                    vTmp = _mm_loadu_ps(add2 +16);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +16),vTmp);

                    aTmp = _mm_loadu_ps(add1 +20);
                    vTmp = _mm_loadu_ps(add2 +20);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +20),vTmp);

                    aTmp = _mm_loadu_ps(add1 +24);
                    vTmp = _mm_loadu_ps(add2 +24);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +24),vTmp);

                    aTmp = _mm_loadu_ps(add1 +28);
                    vTmp = _mm_loadu_ps(add2 +28);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +28),vTmp);
                }
                for (size_t i = (n/32)*32; i < n/16*16; i += 16) {
                    
                    float *add1 = add_half1 + i;
                    float *add2 = add_half2 + i;
                    aTmp = _mm_loadu_ps(add1);
                    vTmp = _mm_loadu_ps(add2);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2),vTmp);

                    aTmp = _mm_loadu_ps(add1 +4);
                    vTmp = _mm_loadu_ps(add2 +4);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +4),vTmp);

                    aTmp = _mm_loadu_ps(add1 +8);
                    vTmp = _mm_loadu_ps(add2+8);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +8),vTmp);

                    aTmp = _mm_loadu_ps(add1 +12);
                    vTmp = _mm_loadu_ps(add2 +12);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add2 +12),vTmp);
                }
                
                for (size_t i = (n/16)*16; i < n/4*4; i += 4) {
                    
                    aTmp = _mm_loadu_ps(add_half1 + i);
                    vTmp = _mm_loadu_ps(add_half2 + i);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((add_half2 + i),vTmp);
                }
                switch (n%4) {
                    case 1: {
                        int i = n/4*4;
                        v[i + ln] += u[j+ln] * A[i + nj];
                        break;
                    }
                    case 2: {
                        int i = n/4*4;
                        v[i + ln] += u[j+ln] * A[i + nj];
                        v[i + ln + 1] += u[j+ln] * A[i + nj + 1];
                        break;
                    }
                    case 3: {
                        int i = n/4*4;
                        v[i + ln] += u[j+ln] * A[i + nj];
                        v[i + ln + 1] += u[j+ln] * A[i + nj + 1];
                        v[i + ln + 2] += u[j+ln] * A[i + nj + 2];
                        break;
                    }
                }
                
            }
        }
        
        float mu[n];
        float tmp[4];
        memset(mu, 0, n * sizeof(float));
        #pragma omp parallel for shared(v, mu) private(vTmp, uTmp,tmp)
        for (size_t l = 0; l < n; l += 1) {
            //__m128  vTmp, uTmp;
            
            memset(tmp, 0, 4 * sizeof(float));
            uTmp = _mm_setzero_ps();
            int ln = l*n;
            for (size_t i = 0; i < n/64*64; i += 64) {
                float *add1 = v + i +ln;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = n/64*64; i < n/32*32; i += 32) {
                float *add1 = v + i +ln;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = n/32*32; i < n/16*16; i+= 16) {
                float *add1 = v + i +ln;
                vTmp = _mm_loadu_ps(add1);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 8);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }

            for (size_t i = n/16*16; i < n/4*4; i+=4) {
                vTmp = _mm_loadu_ps(v + i+ ln);
                uTmp = _mm_add_ps(uTmp, _mm_mul_ps(vTmp, vTmp));
            }
            _mm_storeu_ps(tmp, uTmp);
            mu[l] = tmp[0] +tmp[1] + tmp[2] +tmp[3];
            // broundary case
            for (size_t i = n/4*4; i < n; i++) {
                mu[l] += v[i + ln] * v[i + ln];
            }
            mu[l] = sqrt(mu[l]);
        }

        #pragma omp parallel for shared(v, u) private(vTmp, uTmp, muTmp)
        for (size_t l = 0; l < n; l += 1) {
            muTmp = _mm_set_ps(mu[l] , mu[l], mu[l], mu[l]);
            int ln = l*n;
            for (size_t i = 0; i < n/64*64; i += 64) {
                float *add1 = v +i +ln;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 32);
                _mm_storeu_ps(add2 +32, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 36);
                _mm_storeu_ps(add2 +36, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 40);
                _mm_storeu_ps(add2 +40, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 44);
                _mm_storeu_ps(add2 +44, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 48);
                _mm_storeu_ps(add2 +48, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 52);
                _mm_storeu_ps(add2 +52, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 56);
                _mm_storeu_ps(add2 +56, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 60);
                _mm_storeu_ps(add2 +60, _mm_div_ps(vTmp, muTmp));

            }

            for (size_t i = n/64*64; i < n/32*32; i += 32) {
                float *add1 = v +i +ln;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 16);
                _mm_storeu_ps(add2 +16, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 20);
                _mm_storeu_ps(add2 +20, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 24);
                _mm_storeu_ps(add2 +24, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 28);
                _mm_storeu_ps(add2 +28, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/32*32; i < n/16*16; i +=16){
                float *add1 = v +i +ln;
                float *add2 = u + i +ln;
                vTmp = _mm_loadu_ps(add1);
                _mm_storeu_ps(add2, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 4);
                _mm_storeu_ps(add2 +4, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 +8);
                _mm_storeu_ps(add2 + 8, _mm_div_ps(vTmp, muTmp));

                vTmp = _mm_loadu_ps(add1 + 12);
                _mm_storeu_ps(add2 +12, _mm_div_ps(vTmp, muTmp));
            }

            for (size_t i = n/16*16; i < n/4*4; i +=4) {
                vTmp = _mm_loadu_ps( v + i + ln);
                _mm_storeu_ps( u + i + ln, _mm_div_ps(vTmp, muTmp));
            }
            for (size_t i = n/4*4; i < n; i++) {
                u[i + ln] = v[i + ln] / mu[l];
            }
        }
    }
}

/**
// should not convert to the column.......... WTF...
// OK NO cache block.....
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    
    // try cache block
    int vectorSize = 4; // vector size is 4 *32 bits = 4*4 byte
    int subblockSize = 48; // for each subblock it's 52*52
    size_t N; // it's the size with padding
    float *blockV, *blockA, *blockU, *vectorV, *vectorA, *vectorU;
    __m128 aTmp, vTmp, uTmp;
    register float tmp;


    if(n%subblockSize != 0 || n < subblockSize) {
        N = (n/subblockSize + 1)*subblockSize;
    } else {
        N = n;
    }

    
    float *vReduction = (float*) malloc(4*sizeof(float));

    


    // padding the data
    float *v_p = (float*) malloc(N*N*sizeof(float));
    float *A_p = (float*) malloc(N*N*sizeof(float));
    float *u_p = (float*) malloc(N*N*sizeof(float));

    memset(v_p, 0, sizeof(float) * N * N);
    memset(A_p, 0, sizeof(float) * N * N);
    memset(u_p, 0, sizeof(float) * N * N);

    #pragma omp parallel
    {
        #pragma omp for private(i, j, u, A, u_p, A_p, n, N)

        for (int i = 0; i < n; i++)
        {
            //memcpy((v_p+i*N), (v+i*n), n*sizeof(float));
            memcpy((u_p+i*N), (u+i*n), n*sizeof(float));
            memcpy((A_p+i*N), (A+i*n), n*sizeof(float));
            
        }
    }

    // here begin to calculate
   

    for (size_t k = 0; k < iters; k += 1) {

        memset(v_p, 0, sizeof(float) * N * N);
        for (size_t b_l = 0; b_l < N/subblockSize; b_l += 1) {
            for (size_t b_j = 0; b_j < N/subblockSize; b_j += 1) {
                for (size_t b_i = 0; b_i < N/subblockSize; b_i += 1) {
                    // then multiple the sub matrix here
                    // fisrtly get the position of the start of the each sub matrix
                    blockA = A_p + (b_i + b_l*N)*subblockSize;
                    blockU = u_p + (b_l + b_j*N)*subblockSize;
                    blockV = v_p + (b_i + b_j*N)*subblockSize;

                    for (size_t i = 0; i < subblockSize/vectorSize; i ++) {
                        for (size_t j = 0; j < subblockSize/vectorSize; j++) {
                            vectorV = blockV + (i + j*N)*vectorSize;
                            for (size_t l = 0; l <subblockSize/vectorSize; l++) {
                                //find the star place for the vector matrix (4*4)
                                vectorA = blockA + (i + l*N)*vectorSize;
                                vectorU = blockU + (l + j*N)*vectorSize;
                                
                                //#pragma omp parallel
                                {
                                    //#pragma omp for private(v1, v2, aTmp, uTmp, vectorA, vectorU, vReduction)
                                    //try vector computation
                                    for (size_t vectl = 0; vectl < 4; vectl += 1) {
                                        for (size_t vectj = 0; vectj < 4; vectj += 1) {
                                            tmp = *(vectorU + vectl+vectj*N);

                                            aTmp = _mm_loadu_ps(vectorA + vectl*N);
                                            vTmp = _mm_loadu_ps(vectorV + vectj*N);
                                            uTmp = _mm_set_ps(tmp, tmp, tmp, tmp);
                                            vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                                            _mm_storeu_ps(vectorV + vectj*N,vTmp);
                                            //v[i + l*n] += tmp * A[i + n*j];
                                            
                
                                        }
                                     }
                                    
                                }
                    

                            }
                        }
                    }
                }
            } 
        }


        
        float mu[n];
        memset(mu, 0, n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                mu[l] += v_p[i + l*N] * v_p[i + l*N];
            }
            mu[l] = sqrt(mu[l]);
        }

        
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                u_p[i + l*N] = v_p[i + l*N] / mu[l];
            }
        }
    }
        // convert back to the original one
    #pragma omp parallel
    {
        #pragma omp for private(i, j, v, v_p, n, N)

        for (int i = 0; i < n; i++)
        {
            memcpy((v+i*n), (v_p+i*N), n*sizeof(float));
            
        }
    }
    free(v_p);
    free(u_p);
    free(A_p);
    free(vReduction);

    
}




/**
// no padding edition
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    
    __m128 aTmp, vTmp, uTmp;
    
    float *vReduction = (float*) malloc(4*sizeof(float));
    float *A_p = (float*) malloc(n*n*sizeof(float));
    #pragma omp parallel
    {
        #pragma omp for private(i, j, A, A_p, n)

        for (int i = 0; i < n; i++)
        {
            // then we need to tranpose the A
            for (int j = 0; j < n; j++){
                A_p[j + i*n] = A[i + j*n];
            }
        }
    }

    // here begin to calculate
    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v, 0, n * n * sizeof(float));
        for (size_t i = 0; i < n; i += 1) {
            for (size_t j = 0; j < n; j += 1) {
                for (size_t l = 0; l < n/4 * 4; l += 4) {
                    aTmp = _mm_loadu_ps(A_p + l + i*n);
                    uTmp = _mm_loadu_ps(u + l + j*n);
                    _mm_storeu_ps(vReduction, _mm_mul_ps(uTmp, aTmp));
                    v[i + j*n] += vReduction[0] + vReduction[1] + vReduction[2] +vReduction[3];
                    //v[i + j*N] += A[i*N + l] * u[l + j*N];
                }
                // navie loop
                // edge case
                for (size_t l = (n/4)*4; l < n; l++) {
                    v[i + j*n] += A_p[i*n + l] * u[l + j*n];
                }

            }
        }
        
        float mu[n];
        memset(mu, 0, n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                mu[l] += v[i + l*n] * v[i + l*n];
            }
            mu[l] = sqrt(mu[l]);
        }

        
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                u[i + l*n] = v[i + l*n] / mu[l];
            }
        }
    }



}

**/
/** tranpose and padding...
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    int vectorSize = 4; // vector size is 4 *32 bits = 4*4 byte
    int subblockSize = 4; // for each subblock it's 52*52
    size_t N; // it's the size with padding
    //float *blockV, *blockA, *blockU, *vectorV, *vectorA, *vectorU;
    __m128 aTmp, vTmp, uTmp;


    if(n%subblockSize != 0 || n < subblockSize) {
        N = (n/subblockSize + 1)*subblockSize;
    } else {
        N = n;
    }

    
    float *vReduction = (float*) malloc(4*sizeof(float));

    


    // padding the data
    float *v_p = (float*) malloc(N*N*sizeof(float));
    float *A_p = (float*) malloc(N*N*sizeof(float));
    float *u_p = (float*) malloc(N*N*sizeof(float));

    memset(v_p, 0, sizeof(float) * N * N);
    memset(A_p, 0, sizeof(float) * N * N);
    memset(u_p, 0, sizeof(float) * N * N);

    #pragma omp parallel
    {
        #pragma omp for private(i, j, v, u, A, v_p, u_p, A_p, n, N)

        for (int i = 0; i < n; i++)
        {
            //memcpy((v_p+i*N), (v+i*n), n*sizeof(float));
            memcpy((u_p+i*N), (u+i*n), n*sizeof(float));
            memcpy((A_p+i*N), (A+i*n), n*sizeof(float));
            // then we need to tranpose the A
            for (int j = 0; j < n; j++){
                A_p[j + i*N] = A[i + j*n];
            }
        }
    }

    // here begin to calculate
    for (size_t k = 0; k < iters; k += 1) {
        
        memset(v_p, 0, N * N * sizeof(float));
        for (size_t i = 0; i < N; i += 1) {
            for (size_t j = 0; j < N; j += 1) {
                for (size_t l = 0; l < N; l += 4) {
                    aTmp = _mm_loadu_ps(A_p + l + i*N);
                    uTmp = _mm_loadu_ps(u_p + l + j*N);
                    _mm_storeu_ps(vReduction, _mm_mul_ps(uTmp, aTmp));
                    v_p[i + j*N] += vReduction[0] + vReduction[1] + vReduction[2] +vReduction[3];
                    //v[i + j*N] += A[i*N + l] * u[l + j*N];
                }
            }
        }
        
        float mu[n];
        memset(mu, 0, n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                mu[l] += v_p[i + l*N] * v_p[i + l*N];
            }
            mu[l] = sqrt(mu[l]);
        }

        
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                u_p[i + l*N] = v_p[i + l*N] / mu[l];
            }
        }
    }


    #pragma omp parallel
    {
        #pragma omp for private(i, j, v, v_p, n, N)

        for (int i = 0; i < n; i++)
        {
            memcpy((v+i*n), (v_p+i*N), n*sizeof(float));
            
        }
    }

}



**/


// record for cache blcok......
/**
void eig(float *v, float *A, float *u, size_t n, unsigned iters) {
    
    // try cache block
    int vectorSize = 4; // vector size is 4 *32 bits = 4*4 byte
    int subblockSize = 48; // for each subblock it's 52*52
    size_t N; // it's the size with padding
    //float *blockV, *blockA, *blockU, *vectorV, *vectorA, *vectorU;
    //__m128 aTmp, vTmp, uTmp;


    if(n%subblockSize != 0 || n < subblockSize) {
        N = (n/subblockSize + 1)*subblockSize;
    } else {
        N = n;
    }

    
    float *vReduction = (float*) malloc(4*sizeof(float));

    


    // padding the data
    float *v_p = (float*) malloc(N*N*sizeof(float));
    float *A_p = (float*) malloc(N*N*sizeof(float));
    float *u_p = (float*) malloc(N*N*sizeof(float));

    memset(v_p, 0, sizeof(float) * N * N);
    memset(A_p, 0, sizeof(float) * N * N);
    memset(u_p, 0, sizeof(float) * N * N);

    #pragma omp parallel
    {
        #pragma omp for private( v, u, A, v_p, u_p, A_p, n, N)

        for (int i = 0; i < n; i++)
        {
            //memcpy((v_p+i*N), (v+i*n), n*sizeof(float));
            memcpy((u_p+i*N), (u+i*n), n*sizeof(float));
            memcpy((A_p+i*N), (A+i*n), n*sizeof(float));
            // then we need to tranpose the A
            for (int j = 0; j < n; j++){
                A_p[j + i*N] = A[i + j*n];
            }
        }
    }

    // here begin to calculate
   
    
    
    for (size_t k = 0; k < iters; k += 1) {
        memset(v_p, 0, sizeof(float) * N * N);

        #pragma omp parallel for
        
        
        for (size_t b_j = 0; b_j < N/subblockSize; b_j += 1) {
            for (size_t b_i = 0; b_i < N/subblockSize; b_i += 1) {
                for (size_t b_l = 0; b_l < N/subblockSize; b_l += 1) {
                    // then multiple the sub matrix here
                    // fisrtly get the position of the start of the each sub matrix
                    float *blockV, *blockA, *blockU, *vectorV, *vectorA, *vectorU;
                    __m128 aTmp, vTmp, uTmp;

                    blockA = A_p + (b_l + b_i*N)*subblockSize;
                    blockU = u_p + (b_l + b_j*N)*subblockSize;
                    blockV = v_p + (b_i + b_j*N)*subblockSize;
                    //#pragma omp parallel for private(vectorV, vectorA, vectorU, aTmp, vTmp, uTmp)
                    for (size_t i = 0; i < subblockSize/vectorSize; i ++) {
                        for (size_t j = 0; j < subblockSize/vectorSize; j++) {
                            vectorV = blockV + (i + j*N)*vectorSize;
                            for (size_t l = 0; l <subblockSize/vectorSize; l++) {
                                //find the star place for the vector matrix (4*4)
                                vectorA = blockA + (l + i*N)*vectorSize;
                                vectorU = blockU + (l + j*N)*vectorSize;
                                
                                //#pragma omp parallel
                                {
                                    //#pragma omp for private(v1, v2, aTmp, uTmp, vectorA, vectorU, vReduction)
                                    //try vector computation
                                    for (size_t v1 = 0; v1 < 4; v1++){
                                        for (size_t v2 = 0; v2 < 4; v2++){
                                            aTmp = _mm_loadu_ps(vectorA + v2*N);
                                            uTmp = _mm_loadu_ps(vectorU + v1*N);
                                            _mm_storeu_ps(vReduction, _mm_mul_ps(uTmp, aTmp));
                                            vectorV[v2 + v1*N] += vReduction[0] + vReduction[1] + vReduction[2] +vReduction[3];
                                        }
                                    }
                                }
                    

                            }
                        }
                    }
                }
            } 
        }
        
        

        
        float mu[n];
        memset(mu, 0, n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                mu[l] += v_p[i + l*N] * v_p[i + l*N];
            }
            mu[l] = sqrt(mu[l]);
        }

        
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                u_p[i + l*N] = v_p[i + l*N] / mu[l];
            }
        }
    }
        // convert back to the original one
    #pragma omp parallel
    {
        #pragma omp for private(i, j, v, v_p, n, N)

        for (int i = 0; i < n; i++)
        {
            memcpy((v+i*n), (v_p+i*N), n*sizeof(float));
            
        }
    }
    free(v_p);
    free(u_p);
    free(A_p);
    free(vReduction);

    
}

// record....
/**
        for (size_t l = 0; l < n; l += 1) {
            for (size_t j = 0; j < n; j += 1) {
                tmp = u[j+l*n];

                for (size_t i = 0; i < n/4*4; i += 4) {
                    aTmp = _mm_loadu_ps(&A[n*j+i]);
                    vTmp = _mm_loadu_ps(&v[i + l*n]);
                    uTmp = _mm_set_ps(tmp, tmp, tmp, tmp);
                    vTmp = _mm_add_ps( vTmp, _mm_mul_ps(uTmp, aTmp));
                    _mm_storeu_ps((&v[i + l*n]),vTmp);
                    //v[i + l*n] += tmp * A[i + n*j];
                }
                // edge case
                for (size_t i = (n/4)*4; i < n; i++) {
                    v[i + l*n] += tmp * A[i + n*j];
                }
                
            }
        }
        **/
        // the completely new edition


