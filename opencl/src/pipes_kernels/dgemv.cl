/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    GEMV_V1  performs one of the matrix-vector operations

    -   y := alpha*A*x + beta*y,
            where the NxM matrix A is received in tiles streamed by rows, where each
            tile is Row Streamed. x is an M-elements vector, while y
            is an N-element vector

    -   or  y := alpha*A**T*x + beta*y,
            where the NxM matrix A is received in tiles streamed by columns,
            each tile is Column Streamed. x is an N-element vector, while y
            is an M-element vector

    Data is received from three different channels (channel_in_vector_x_1, channel_in_vector_y_1
    and channel_in_matrix_A_1). Input data must be padded with zeros according to
    the reference tile sizes (TILE_N and TILE_M).

    Result is streamed in an output channel as soon as it is available.

    Check the kernel documentation for further information

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable


channel double channel_in_vector_x_1 __attribute__((depth(8)));
channel double channel_in_vector_y_1 __attribute__((depth(8)));
channel double channel_in_matrix_A_1 __attribute__((depth(8)));
channel double channel_out_vector_1 __attribute__((depth(1)));

/**
    This version is meant for the following cases:
    - A is RowStreamed and NonTransposed, Tiles received by rows. In this case:
            - row_streamed must be set to 1
            - x is a vector of M elements, y is a vector of N elements
            - blocks of TILE_N elements of y are reused
            - also block of TILE_M elements of x are reused. The entire vector x must be resent N/TILE_N times (i.e. len_y/tile_y)
    - A is ColStreamed and Transposed, Tiles received by cols:
            - row_streamed must be set to 0
            - x is a vector of N elements, while y is a vector of M elements
            - blocks of y are of TILE_M elements
            - the entire x must be resent M/TILE_M times. Reuse will be applied also to it

    Matrix and vector must be padded to the respective tiling sizes
*/
__kernel void dgemv(int row_streamed, const int N, const int M, const double alpha, const double beta)
{
    __constant uint WIDTH = 8;
    __constant uint TILE_N = 256;
    __constant uint TILE_M = 256;
    __constant uint MAX_TILE_SIZE = 256;

    __constant uint SHIFT_REG = 64;

    int len_x,tile_x;
    int len_y,tile_y;
    int BlocksX, BlocksY;
    //chose the loop limits
    if(row_streamed == 1)
    {
        len_x = M;
        len_y = N;
        tile_x=TILE_M;
        tile_y=TILE_N;
        BlocksY=1+(int)((N-1)/TILE_N); //ceiling
        BlocksX=1+(int)((M-1)/TILE_M);
    }
    else
    {	//in this case A is transposed
        len_x = N;
        len_y = M;
        tile_x=TILE_N;
        tile_y=TILE_M;
        BlocksY=1+(int)((M-1)/TILE_M);
        BlocksX=1+(int)((N-1)/TILE_N);
    }

    double shift_reg[SHIFT_REG+1]; //shift register

    #pragma unroll
    for(int i=0;i<SHIFT_REG+1;i++)
       shift_reg[i]=0;


    //The computation is performed by receiving A in tiles by row (A non transposed) or column (A transposed).
    //In this way, the result is computed by 'accumulating' over y elements
    //One block of y is computed for each row-tile (or column-tile) of A and using the entire x

    const int computing_outer_loop_limit=(int)(tile_x/WIDTH);
    const int reading_y_outer_loop_limit=(int)(tile_y/WIDTH);

    double local_y[MAX_TILE_SIZE];
    double local_x[MAX_TILE_SIZE];

    //Please note: the order in which tiles arrive, will determine the computation
    //(i.e. do not assume that you will receive the tiles one row after the other...maybe they can arrive column by column)

    #pragma loop_coalesce
    #pragma ivdep
    for(int ti=0;ti<BlocksY;ti++)
    {
        #pragma ivdep
        for(int tj=0;tj<BlocksX;tj++)
        {
            //To buffer x, we will use the first iteration of the main loop
            //Also here, do not be confused by i and j, they can refer to rows and column of columns and rows

            #pragma ivdep
            for(int i=0;i<tile_y;i++)
            {

                double prev;

                //here we read one element from A and one element from X and we use it
                //For X we buffer it at the first iteration
                //this should not be a problem if tile_y is distant
                double acc_o=0;

                #pragma unroll
                for(int i=0;i<SHIFT_REG+1;i++)
                    shift_reg[i]=0;

                #pragma ivdep
                for(int jj=0;jj<computing_outer_loop_limit;jj++)
                {

                    if(tj==0 && jj==0)//put here to have evertyhing in the loop
                    {
                        if(beta==0)
                            prev=0;
                        else
                           prev=beta*read_channel_intel(channel_in_vector_y_1);
                    }

                    if(i==0)
                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                           local_x[jj*WIDTH+j]=read_channel_intel(channel_in_vector_x_1);

                    double acc_i=0;
                    //read (a block of W elements) of the row of A
                    double local_A[WIDTH];
                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                        local_A[j]=read_channel_intel(channel_in_matrix_A_1);

                    #pragma unroll
                    for(int j=0;j<WIDTH;j++)
                        acc_i+=local_A[j]*local_x[jj*WIDTH+j];

                    shift_reg[SHIFT_REG] = shift_reg[0]+alpha*acc_i;
                    //Shift every element of shift register
                    #pragma unroll
                    for(int j = 0; j < SHIFT_REG; ++j)
                        shift_reg[j] = shift_reg[j + 1];

                    acc_o=0;
                    #pragma unroll
                    for(int i=0;i<SHIFT_REG;i++)
                    {
                        acc_o+=shift_reg[i];
                    }
                    if(jj == computing_outer_loop_limit -1){
                    //         //no beta version
                        if(tj!=0)
                            prev=local_y[i];
                        double result =  prev+  acc_o;
                        local_y[i] = result;
                        //output y if we reached the end of the matrix
                        //y is output one element at a time
                        if(tj==BlocksX-1)
                          write_channel_intel(channel_out_vector_1,result);
                    }
                }

            }
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into channel_in_vector_x_1. The vector is accessed with stride 1.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/


__kernel void kernel_read_vector_x_1(__global volatile double  *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    __constant uint WIDTH = 8;
    __constant int INCX = 1;

    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    #pragma loop_coalesce
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCX) > 0 ?  0 : ((N) - 1) * (-(INCX)));

        for(int i=0;i<outer_loop_limit;i++)
        {
            double x[WIDTH];
            //prepare data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
            {
                if(i*WIDTH+k<N)
                    x[k]=data[offset+(k*INCX)];
                else
                    x[k]=0;
            }
            offset+=WIDTH*INCX;

            //send data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
                write_channel_intel(channel_in_vector_x_1,x[k]);
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type double from memory and  push it
    into channel_in_vector_y_1. The vector is accessed with 1 INCY.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded to W using zero elements.

    The vector is sent 'repetitions' times.
*/


__kernel void kernel_read_vector_y_1(__global volatile double *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
{
    __constant uint WIDTH = 8;
    __constant int INCY = 1;

    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;

    #pragma loop_coalesce
    for(int t=0; t< repetitions;t++)
    {
        //compute the starting index
        int offset=((INCY) > 0 ?  0 : ((N) - 1) * (-(INCY)));
        for(int i=0;i<outer_loop_limit;i++)
        {
            double y[WIDTH];
            //prepare data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
            {
                if(i*WIDTH+k<N)
                    y[k]=data[offset+(k*INCY)];
                else
                    y[k]=0;
            }
            offset+=WIDTH*INCY;

            //send data
            #pragma unroll
            for(int k=0;k<WIDTH;k++)
                write_channel_intel(channel_in_vector_y_1,y[k]);
        }
    }
}
/**

    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a matrix of type double from memory and  push it
    into channel_in_matrix_A_1. The matrix is sent RowStreamed (i.e. one row after another)
    and Tiles are sent by row. Tiles have size 256 x 256.


    8 reads are performed simultaneously.
    If needed, data is padded to tile sizes using zero elements.


*/



__kernel void kernel_read_matrix_A_1(__global double *restrict data, int N, int M, unsigned int lda)
{

    __constant uint WIDTH = 8;
    __constant uint TILE_N = 256;
    __constant uint TILE_M = 256;
    const int BlocksN=1+(int)((N-1)/TILE_N);
    const int BlocksM=1+(int)((M-1)/TILE_M);
    const int outer_loop_limit=((int)(TILE_M))/WIDTH;   //WIDTH must be a divisor of TILE
    #pragma loop_coalesce
    for(int ti=0;ti<BlocksN;ti++)
    {
        for(int tj=0;tj<BlocksM;tj++)
        {
            for(int i = 0; i < TILE_N; i++)
            {
                for(int j=0; j < outer_loop_limit; j++ )
                {
                    double to_send[WIDTH];
                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                    {
                        if((ti*TILE_N+i)<N  && tj*TILE_M+j*WIDTH+jj< M)
                            to_send[jj] = data[(ti*TILE_N+i)*lda+tj*TILE_M+j*WIDTH+jj];
                        else
                            to_send[jj]=0;
                    }

                    #pragma unroll
                    for(int jj = 0; jj < WIDTH; jj++)
                        write_channel_intel(channel_in_matrix_A_1,to_send[jj]);

                }
            }
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Write a vector of type double  into  memory.
    The vector elements are read from channel channel_out_vector_1.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCW represent the access stride.

    WIDTH reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.
*/


__kernel void kernel_write_vector_1(__global volatile double  *restrict out, unsigned int N, unsigned int pad_size)
{
    __constant uint WIDTH = 8;
    __constant int INCW = 1;
    const unsigned int ratio=pad_size/WIDTH;
    const unsigned int padding_loop_limit=ceil(((float)N)/pad_size);
    const unsigned int outer_loop_limit=padding_loop_limit*ratio;
    double  recv[WIDTH];
    //compute the starting index
    int offset=((INCW) > 0 ?  0 : ((N) - 1) * (-(INCW)));
    //receive and store data into memory
    for(int i=0;i<outer_loop_limit;i++)
    {
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
        {
            recv[j]=read_channel_intel(channel_out_vector_1);

            if(i*WIDTH+j<N)
                out[offset+(j*INCW)]=recv[j];
        }
        offset+=WIDTH*INCW;
    }
}