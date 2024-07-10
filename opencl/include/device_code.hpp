#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP 

#include "CL/cl.hpp"
#include "utils.hpp"
// oneAPI headers

class PipesDeviceHandler {
    public:
        PipesDeviceHandler(cl::Device &_device, cl::Context &_context, cl::CommandQueue &_queue, cl::Program &_program, cl::Program &_fblas_program) : 
            device(_device),
            context(_context),
            queue(_queue),
            program(_program),
            fblas_program(_fblas_program)
            {};

        void dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size);
        void vec_sum(double alpha, const cl::Buffer &bufa, double beta, const cl::Buffer &bufB, size_t size);
        void matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &dB, const cl::Buffer &bufC, size_t size);

    private:
        cl::Device device;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;
        cl::Program fblas_program;
        

};

class DirectDeviceHandler {
    public:
        DirectDeviceHandler(cl::Device &_device, cl::Context &_context, cl::CommandQueue &_queue, cl::Program &_program, cl::Program &_fblas_program) : 
            device(_device),
            context(_context),
            queue(_queue),
            program(_program),
            fblas_program(_fblas_program)
            {};

        void dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size);
        void vec_sum(double alpha, const cl::Buffer &bufa, double beta, const cl::Buffer &bufB, size_t size);
        void matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &dB, const cl::Buffer &bufC, size_t size);

    private:
        cl::Device device;
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;
        cl::Program fblas_program;
        

};



#endif