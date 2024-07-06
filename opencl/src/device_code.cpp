#include "device_code.hpp"
 

void PipesDeviceHandler::dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, std::vector<cl::Event> &events) {


    cl_int err; 
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(4, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_x_0", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_y_0", &err));
    kernels.push_back(cl::Kernel(fblas_program, "ddot", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_write_scalar_0", &err));
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
void PipesDeviceHandler::vec_sum(double alpha, const cl::Buffer &bufA, double beta, const cl::Buffer &bufB, size_t size, cl::Event &event){
    /*
    cl_int err;
    cl::Kernel kernel(program, "VecSum", &err);
    CHECK_ERR(err, "Failed to create VecSum kernel");


    unsigned _size = size; 

    err = kernel.setArg(0, alpha);
    err |= kernel.setArg(1,bufX);
    err |= kernel.setArg(2, beta);
    err |= kernel.setArg(3, bufY);
    err |= kernel.setArg(4, sizeof(unsigned), &_size);
    CHECK_ERR(err, "Failed to set vec sum kernel arguments");

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);              
    CHECK_ERR(err, "Failed to enqueue VecSum kernel");
*/

    cl_int err; 
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(4, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_x_2", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_y_2", &err));
    kernels.push_back(cl::Kernel(fblas_program, "daxpy", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_write_vector_2", &err));
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

void PipesDeviceHandler::matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, cl::Event &event){
/*    
    cl::Kernel kernel(fblas_program, "GEMV", &err);
    CHECK_ERR(err, "Failed to create GEMV kernel");


    err = kernel.setArg(0, bufA);
    err |= kernel.setArg(1, bufB);
    err |= kernel.setArg(2, bufC);
    err |= kernel.setArg(3, sizeof(unsigned), &size);
    CHECK_ERR(err, "Failed to set gemv kernel arguments");

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, NULL, &event);              
    CHECK_ERR(err, "Failed to enqueue Gemv kernel.");
    queue.finish();
    return;
*/

    cl_int err;
    std::vector<cl::Kernel> kernels;
    std::vector<cl::CommandQueue> queues(5, std::move(cl::CommandQueue(context)));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_matrix_A_1", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_x_1", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_read_vector_y_1", &err));
    kernels.push_back(cl::Kernel(fblas_program, "dgemv", &err));
    kernels.push_back(cl::Kernel(fblas_program, "kernel_write_vector_1", &err));
    CHECK_ERR(err, "Error gemv fblas kernel creation");

    
    const int _size = size;
    const unsigned int width = 8;
    const unsigned int one = 1;
    const unsigned int zero = 0;
    const unsigned int order = 1;
    const int tile = 256;
    const double alpha = 1.0;
    const double beta = 0.0;
    const uint x_repetitions=ceil((float)(size)/tile);

//    FBLASEnvironment::gemv<double>(routine_name,transposed,N,M,alpha,A,lda,x,incx,beta,y,incy,events_wait_list,event);
    //fb.dgemv("dgemv", FBLAS_NO_TRANSPOSED, size, size, 1.0, fpga_A, size, fpga_p, 1, 0.0, fpga_Ap, 1);

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
