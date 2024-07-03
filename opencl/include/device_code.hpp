#ifndef DEVICE_CODE_HPP
#define DEVICE_CODE_HPP 

#include "CL/cl.hpp"
#include "utils.hpp"
// oneAPI headers

class Device {
    public:
        Device(cl::Context &_context, cl::CommandQueue &_queue, cl::Program &_program) : 
            context(_context),
            queue(_queue),
            program(_program){};

        void dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, cl::Event &event);
        void vec_sum(double alpha, const cl::Buffer &bufX, double beta, const cl::Buffer &bufY, size_t size, cl::Event &event);
        void matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &dB, const cl::Buffer &bufC, size_t size, cl::Event &event);

    private:
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;
        

};


#endif