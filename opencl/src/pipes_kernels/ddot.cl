 /**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    DOT performs the dot product of two vectors.

    Streamed version: data is received from two input streams
    channel_in_vector_x_0 and channel_in_vector_y_0 having the proper type.
    Data elements must be streamed with a padding equal to W
    (padding data must be set to zero).

    Result is streamed in an output channel at the end of the computation
    in a channel channel_out_scalar_0

*/

#pragma OPENCL EXTENSION cl_intel_channels : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable


channel double channel_in_vector_x_0 __attribute__((depth(4096)));
channel double channel_in_vector_y_0 __attribute__((depth(4096)));
channel double channel_out_scalar_0 __attribute__((depth(1)));


/**
    Performs streaming dot product: data is received through
    channel_in_vector_x_0 and channel_in_vector_y_0. Result is sent
    to channel_out_scalar_0.
*/
__kernel void ddot(int N)
{
    __constant uint WIDTH = 8;
    __constant uint SHIFT_REG = 64;


    double acc_o=0;
    if(N>0)
    {

        const int outer_loop_limit=1+(int)((N-1)/WIDTH); //ceiling
        double x[WIDTH],y[WIDTH];

        double shift_reg[SHIFT_REG+1]; //shift register

        #pragma unroll
        for(int i=0;i<SHIFT_REG+1;i++)
           shift_reg[i]=0;

        //Strip mine the computation loop to exploit unrolling
        for(int i=0; i<outer_loop_limit; i++)
        {

            double acc_i=0;
            #pragma unroll
            for(int j=0;j<WIDTH;j++)
            {
                x[j]=read_channel_intel(channel_in_vector_x_0);
                y[j]=read_channel_intel(channel_in_vector_y_0);
                acc_i+=x[j]*y[j];

            }


                shift_reg[SHIFT_REG] = shift_reg[0]+acc_i;
                //Shift every element of shift register
                #pragma unroll
                for(int j = 0; j < SHIFT_REG; ++j)
                    shift_reg[j] = shift_reg[j + 1];

        }

            //reconstruct the result using the partial results in shift register
            double acc=0;
            #pragma unroll
            for(int i=0;i<SHIFT_REG;i++)
                acc+=shift_reg[i];
            acc_o = acc;
    }
    else //no computation: result is zero
        acc_o=0.0f;
    //write to the sink
    write_channel_intel(channel_out_scalar_0,acc_o);
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type TYPE_T from memory and  push it
    into channel_in_vector_x_0. The vector is accessed with stride 1.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded using zero elements.

    The vector is sent 'repetitions' times
*/


__kernel void kernel_read_vector_x_0(__global volatile double  * data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
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
                write_channel_intel(channel_in_vector_x_0,x[k]);
        }
    }
    
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2019 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a vector of type double from memory and  push it
    into channel_in_vector_y_0. The vector is accessed with 1 INCY.
    The name of the kernel can be redefined by means of preprocessor MACROS.

    W=8 memory reads are performed simultaneously. In the same way W channel push are performed.
    Data is padded to pad_size. Pad_size must be a multiple of W.
    So, for level 1 routines pad_size will be probably equal to W.
    For level 2, in which you have to respect some tiling, it will be equal to a tile size.

    Data is padded to W using zero elements.

    The vector is sent 'repetitions' times.
*/


__kernel void kernel_read_vector_y_0(__global volatile double * data, unsigned int N, unsigned int pad_size, unsigned int repetitions)
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
                write_channel_intel(channel_in_vector_y_0,y[k]);
        }
    }
    
}
/**
    FBLAS: BLAS implementation for Intel FPGA
    Copyright (c) 2020 ETH-Zurich. All rights reserved.
    See LICENSE for license information.

    Reads a scalar vector of type double  and writes it into memory

*/


__kernel void kernel_write_scalar_0(__global volatile double * out)
{
    *out = read_channel_intel(channel_out_scalar_0);
}