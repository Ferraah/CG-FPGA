#include "device_code.hpp"
 

void DirectDeviceHandler::dot(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, std::vector<cl::Event> &events) {


    cl_int err; 
    cl::Kernel kernel(fblas_program, "ddot", &err);
    CHECK_ERR(err, "Error dot fblas kernel creation");

    
    const unsigned int width = 8;
    const unsigned int one = 1;
    const int _size = size; 

    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, sizeof(unsigned int), &size);

    // Single tasks
   
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in dot enqueue.");

    queue.finish();
    

}
void DirectDeviceHandler::vec_sum(double alpha, const cl::Buffer &bufA, double beta, const cl::Buffer &bufB, size_t size, cl::Event &event){
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
    cl::Kernel kernel (fblas_program, "daxpy", &err);
    CHECK_ERR(err, "Error dot fblas kernel creation");


    const int _size = size;
    const unsigned int width = 8;
    const unsigned int one = 1;
    

    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufB);
    kernel.setArg(3, sizeof(double), &alpha);
    kernel.setArg(4, sizeof(double), &beta);
    kernel.setArg(5, sizeof(int), &_size);

   
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in vec_sum enqueue.");

    queue.finish();

   
}

void DirectDeviceHandler::matrix_vector_mul(const cl::Buffer &bufA, const cl::Buffer &bufB, const cl::Buffer &bufC, size_t size, cl::Event &event){
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
    cl::Kernel kernel(cl::Kernel(fblas_program, "dgemv", &err));
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
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);
    kernel.setArg(3, sizeof(unsigned),&one);
    kernel.setArg(4, sizeof(int),&_size);
    kernel.setArg(5, sizeof(int),&_size);
    kernel.setArg(6, sizeof(double),&alpha);
    kernel.setArg(7, sizeof(double),&beta);


    // Single tasks
   
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1), cl::NullRange); 
    CHECK_ERR(err, "Error in gemv enqueue.");

    queue.finish();
  
}
