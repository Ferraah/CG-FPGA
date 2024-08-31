/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    AXPY constant times a vector plus a vector.

    Streamed version: data is received from two input streams
    channel_in_vector_x_2 and channel_in_vector_y_2 having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel channel_out_vector_2

*/


#pragma OPENCL EXTENSION cl_intel_channels : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable


channel double channel_in_vector_x_2 __attribute__((depth(8)));
channel double channel_in_vector_y_2 __attribute__((depth(8)));
channel double channel_out_vector_2 __attribute__((depth(8)));

__kernel void daxpy(const double  alpha, int N)
{
    __constant uint WIDTH = 8;
    if(N==0) return;

    const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
    double  res[WIDTH];

    for(int i=0; i<outer_loop_limit; i++)
    {
        //receive WIDTH elements from the input channels
        #pragma unroll
        for(int j=0;j<WIDTH;j++)
            res[j]=alpha*read_channel_intel(channel_in_vector_x_2)+read_channel_intel(channel_in_vector_y_2);

        //sends the data to a writer
        #pragma unroll
        for(int j=0; j<WIDTH; j++)
            write_channel_intel(channel_out_vector_2,res[j]);
    }

}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into channel_in_vector_x_2. The vector is accessed with stride 1.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/


__kernel void kernel_read_vector_x_2(__global volatile double  *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
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
                write_channel_intel(channel_in_vector_x_2,x[k]);
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type double from memory and  push it
    into channel_in_vector_y_2. The vector is accessed with 1 INCY.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded to W using zero elements.

    The vector is sent 'repetitions' times.
*/


__kernel void kernel_read_vector_y_2(__global volatile double *restrict data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
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
                write_channel_intel(channel_in_vector_y_2,y[k]);
        }
    }
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.


    Write a vector of type double  into  memory.
    The vector elements are read from channel channel_out_vector_2.
    The name of the kernel can be redefined by means of preprocessor MACROS.
    INCW represent the access stride.

    WIDTH reads are performed simultaneously.
    Data arrives padded at pad_size.
    Padding data (if present) is discarded.
*/


__kernel void kernel_write_vector_2(__global volatile double  *restrict out, unsigned int N, unsigned int pad_size)
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
            recv[j]=read_channel_intel(channel_out_vector_2);

            if(i*WIDTH+j<N)
                out[offset+(j*INCW)]=recv[j];
        }
        offset+=WIDTH*INCW;
    }
}