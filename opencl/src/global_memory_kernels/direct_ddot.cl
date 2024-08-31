#pragma OPENCL EXTENSION cl_khr_fp64 : enable 

//#define EMULATOR


__kernel void ddot(__global double* vector_x, __global double* vector_y, __global double* restrict out, int N) {
    __constant uint SHIFT_REG = 64;

    double shift_reg[SHIFT_REG+1];

    #pragma unroll
    for (int i = 0; i < SHIFT_REG + 1; i++) {
        shift_reg[i] = 0;
    }

    #pragma ivdep
    for(int i=0; i< N; i++){
        shift_reg[SHIFT_REG] = shift_reg[0] + vector_x[i]*vector_y[i];

        #pragma unroll
        for(int j=0;j<SHIFT_REG;j++){
            shift_reg[j]=shift_reg[j+1];
        }
    }

    double temp_sum = 0; 
    #pragma unroll
    for (int i = 0; i < SHIFT_REG; i++) {
        temp_sum += shift_reg[i];
    }

    out[0] = temp_sum;
}

