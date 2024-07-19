#pragma OPENCL EXTENSION cl_khr_fp64 : enable 

//#define EMULATOR



__kernel void dgemv( __global const double*  restrict A, __global const double* restrict x, __global double* restrict y, const int row_streamed, const int N, const int M, const double alpha, const double beta) {
    __constant int TILE_N = 256;
    __constant int TILE_M = 256;
    //__constant int SHIFT_REG = 64;
    __constant int SHIFT_REG = 128;

    int len_x, tile_x;
    int len_y, tile_y;
    int BlocksX, BlocksY;

    if (row_streamed == 1) {
        len_x = M;
        len_y = N;
        tile_x = TILE_M;
        tile_y = TILE_N;
        BlocksY = 1 + (int)((N - 1) / TILE_N);
        BlocksX = 1 + (int)((M - 1) / TILE_M);
    } else {
        len_x = N;
        len_y = M;
        tile_x = TILE_N;
        tile_y = TILE_M;
        BlocksY = 1 + (int)((M - 1) / TILE_M);
        BlocksX = 1 + (int)((N - 1) / TILE_N);
    }

    double shift_reg[SHIFT_REG + 1];

    #pragma unroll
    for (int i = 0; i < SHIFT_REG + 1; i++)
        shift_reg[i] = 0;

    const int computing_outer_loop_limit = (int)(tile_x / SHIFT_REG);

    __local double local_y[TILE_N];
    __local double local_x[TILE_M];

    //#pragma loop_coalesce
    for (int ti = 0; ti < BlocksY; ti++) {
        //#pragma ivdep
        for (int tj = 0; tj < BlocksX; tj++) {
            //#pragma ivdep
            for (int i = 0; i < tile_y; i++) {
                double prev;
                double acc_o = 0;

                #pragma unroll
                for (int ii = 0; ii < SHIFT_REG + 1; ii++)
                    shift_reg[ii] = 0;

                //#pragma ivdep
                for (int jj = 0; jj < computing_outer_loop_limit; jj++) {
                    if (tj == 0 && jj == 0) {
                        if (beta == 0)
                            prev = 0;
                        else
                            prev = beta * y[ti * TILE_N + i];
                    }

                    if (i == 0) {
                        #pragma unroll
                        for (int j = 0; j < SHIFT_REG; j++) {
                            if (tj * TILE_M + jj * SHIFT_REG + j < len_x) {
                                local_x[jj * SHIFT_REG + j] = x[tj * TILE_M + jj * SHIFT_REG + j];
                            } else {
                                local_x[jj * SHIFT_REG + j] = 0;
                            }
                        }
                    }

                    double acc_i = 0;
                    double local_A[SHIFT_REG];

                    #pragma unroll
                    for (int j = 0; j < SHIFT_REG; j++) {
                        if (ti * TILE_N + i < len_y && tj * TILE_M + jj * SHIFT_REG + j < len_x) {
                            local_A[j] = A[(ti * TILE_N + i) * len_x + tj * TILE_M + jj * SHIFT_REG + j];
                        } else {
                            local_A[j] = 0;
                        }
                    }

                    #pragma unroll
                    for (int j = 0; j < SHIFT_REG; j++) {
                        acc_i += local_A[j] * local_x[jj * SHIFT_REG + j];
                    }

                    shift_reg[SHIFT_REG] = shift_reg[0] + alpha * acc_i;

                    #pragma unroll
                    for (int j = 0; j < SHIFT_REG; ++j) {
                        shift_reg[j] = shift_reg[j + 1];
                    }

                    acc_o = 0;
                    #pragma unroll
                    for (int ii = 0; ii < SHIFT_REG; ii++) {
                        acc_o += shift_reg[ii];
                    }

                    if (jj == computing_outer_loop_limit - 1) {
                        if (tj != 0) {
                            prev = local_y[i];
                        }
                        double result = prev + acc_o;
                        local_y[i] = result;

                        if (tj == BlocksX - 1) {
                            if (ti * TILE_N + i < len_y) {
                                y[ti * TILE_N + i] = result;
                            }
                        }
                    }
                }
            }
        }
    }
}

