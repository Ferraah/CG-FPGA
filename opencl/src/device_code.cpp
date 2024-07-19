#include "device_code.hpp"
 

void PipesDeviceHandler::dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size) {

    cl_int err; 
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(4, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_x_0", &err));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_y_0", &err));
    kernels.push_back(cl::Kernel(program, "ddot", &err));
    kernels.push_back(cl::Kernel(program, "kernel_write_scalar_0", &err));
    CHECK_ERR(err, "Error dot fblas kernel creation");

    
    const unsigned int width = 8;
    const unsigned int one = 1;
    const int _size = size; 

    kernels[0].setArg(0, bufA);
    kernels[0].setArg(1, sizeof(unsigned int), &size);
    kernels[0].setArg(2, sizeof(unsigned int), &width);
    kernels[0].setArg(3, sizeof(unsigned int), &one);

    kernels[1].setArg(0, bufB);
    kernels[1].setArg(1, sizeof(unsigned int), &size);
    kernels[1].setArg(2, sizeof(unsigned int),&width);
    kernels[1].setArg(3, sizeof(unsigned int),&one);

    kernels[2].setArg(0, sizeof(int), &_size);

    kernels[3].setArg(0, bufC);

    // Single tasks
   
    for(uint i = 0; i<4; i++){
        err = queues[i].enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(1), cl::NullRange); 
        CHECK_ERR(err, "Error in dot enqueue.");
    }

    queues[3].finish();
    

}
void PipesDeviceHandler::vec_sum(double alpha, const cl::Buffer &bufA, double beta, const cl::Buffer &bufB, size_t size){

    cl_int err; 
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(4, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_x_2", &err));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_y_2", &err));
    kernels.push_back(cl::Kernel(program, "daxpy", &err));
    kernels.push_back(cl::Kernel(program, "kernel_write_vector_2", &err));
    CHECK_ERR(err, "Error dot fblas kernel creation");


    const int _size = size;
    const unsigned int width = 8;
    const unsigned int one = 1;
    

    kernels[0].setArg(0, bufA);
    kernels[0].setArg(1,sizeof(int), &_size);
    kernels[0].setArg(2, sizeof(unsigned int), &width);
    kernels[0].setArg(3, sizeof(unsigned int), &one);

    kernels[1].setArg(0, bufB);
    kernels[1].setArg(1,sizeof(int), &_size);
    kernels[1].setArg(2, sizeof(unsigned int),&width);
    kernels[1].setArg(3, sizeof(unsigned int),&one);

    kernels[2].setArg(0, sizeof(double), &alpha);
    kernels[2].setArg(1, sizeof(double), &beta);
    kernels[2].setArg(2, sizeof(int), &_size);

    kernels[3].setArg(0, bufB);
    kernels[3].setArg(1,sizeof(int), &_size);
    kernels[3].setArg(2, sizeof(unsigned int),&width);

   
    for(uint i = 0; i<4; i++){
        err = queues[i].enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(1), cl::NullRange); 
        CHECK_ERR(err, "Error in vec_sum enqueue.");
    }

    queues[3].finish();

   
}

void PipesDeviceHandler::matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size){

    cl_int err;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(5, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(program, "kernel_read_matrix_A_1", &err));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_x_1", &err));
    kernels.push_back(cl::Kernel(program, "kernel_read_vector_y_1", &err));
    kernels.push_back(cl::Kernel(program, "dgemv", &err));
    kernels.push_back(cl::Kernel(program, "kernel_write_vector_1", &err));
    CHECK_ERR(err, "Error gemv fblas kernel creation");

    
    const int _size = size;
    const unsigned order = 1;
    const int tile = 256;
    const double alpha = 1.0;
    const double beta = 0.0;
    const unsigned zero = 0.0;
    const uint x_repetitions=ceil((float)(size)/tile);

    //matrix reader
    kernels[0].setArg(0, bufA);
    kernels[0].setArg(1, sizeof(int),&size);
    kernels[0].setArg(2, sizeof(int),&size);
    kernels[0].setArg(3, sizeof(int),&size);

    kernels[1].setArg(0, bufB);
    kernels[1].setArg(1, sizeof(unsigned int),&size);
    kernels[1].setArg(2, sizeof(unsigned int),&tile);
    kernels[1].setArg(3, sizeof(unsigned int),&x_repetitions);

    // Useless 
    kernels[2].setArg(0, bufC);
    kernels[2].setArg(1, sizeof(unsigned int),&size);
    kernels[2].setArg(2, sizeof(unsigned int),&tile);
    kernels[2].setArg(3, sizeof(unsigned int),&zero);


    kernels[3].setArg(0, sizeof(int),&order);
    kernels[3].setArg(1, sizeof(int),&_size);
    kernels[3].setArg(2, sizeof(int),&_size);
    kernels[3].setArg(3, sizeof(double),&alpha);
    kernels[3].setArg(4, sizeof(double),&beta);

    kernels[4].setArg(0, bufC);
    kernels[4].setArg(1, sizeof(unsigned int),&size);
    kernels[4].setArg(2, sizeof(unsigned int),&tile);


    // Single tasks
   
    for(uint i = 0; i<kernels.size(); i++){
        err = queues[i].enqueueNDRangeKernel(kernels[i], cl::NullRange, cl::NDRange(1), cl::NullRange); 
        CHECK_ERR(err, "Error in gemv enqueue."+std::to_string(i));
    }

    queues[kernels.size()-1].finish();
  
}
