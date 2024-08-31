#pragma OPENCL EXTENSION cl_khr_fp64 : enable 

//#define EMULATOR

/*
__kernel void daxpy(__global restrict double* vector_x, __global double* vector_y, __global double* out, const double alpha, const double beta, int N) {

    if (N == 0) return;

    __local double x, y; 
    for (int i = 0; i < N; i++) {
        x = vector_x[i];
        y = vector_y[i];
        out[i] = alpha * x  + beta * y;
    }
}
*/


#define BLOCK_SIZE 512 

__kernel
void daxpy(
__global const double *restrict x,
__global double *y,
__global double *out,
const double alpha,
const double beta,
const int size)
{
    #pragma ivdep
    for (int i=0; i<size; i+=BLOCK_SIZE){
        float local_x[BLOCK_SIZE];
        float local_y[BLOCK_SIZE];
        
        #pragma unroll 16
        for (int j=0; j<BLOCK_SIZE; j++){
            local_x[j] = x[i+j];
        }
        #pragma unroll 16
        for (int j=0; j<BLOCK_SIZE; j++){
            local_y[j] = y[i+j];
        }
        #pragma unroll 16
        for (int j=0; j<BLOCK_SIZE; j++){
            out[i+j] = alpha*local_x[j] + beta*local_y[j];
        }
    }
}