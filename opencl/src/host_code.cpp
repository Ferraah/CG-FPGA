#include "host_code.hpp"
#include <chrono>
#include "sequential.hpp"
#include <mpi.h>


#define PLATFORM_ID 2
#define N_FPGA 2

void prepare(cl::Context &context, cl::CommandQueue &queue, cl::Program &program, int fpga_id){

    cl_int err = CL_SUCCESS;
    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> platform_devices;

    // Get all the platforms available
    cl::Platform::get(&all_platforms);

    // Usually FPGA platforms are the last ones on meluxina 
    err = all_platforms[PLATFORM_ID].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    CHECK_ERR(err, "Failed to get platform ID");

    // Create the context for the devices
    context = cl::Context(platform_devices[fpga_id]);

    // Add the platform devices to the context

    //std::cout << "Using device: " << platform_devices[fpga_id].getInfo<CL_DEVICE_NAME>() << "\n";

    // Create queue and program for each device
    queue = cl::CommandQueue(context, platform_devices[fpga_id], 0, &err);
    CHECK_ERR(err, "Failed to create command queue");


    utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/bin/global_memory_kernels.aocx", program, context, platform_devices[fpga_id]);


}


void prepare(cl::Context &context, cl::CommandQueue &q,cl::Program &program){

    cl_int err = CL_SUCCESS;
    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> platform_devices;
    std::vector<cl::Device> devices;

    // Get all the platforms available
    cl::Platform::get(&all_platforms);

    // Usually FPGA platforms are the last ones on meluxina 
    err = all_platforms[PLATFORM_ID].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    CHECK_ERR(err, "Failed to get platform ID");

    auto device = platform_devices[0];

    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    context = cl::Context(device);
    CHECK_ERR(err, "Failed to create context");

    q = cl::CommandQueue(context, device, 0, &err);

    CHECK_ERR(err, "Failed to create command queue");

    // @TODO: Make a unique file for emulation kernel
//    utils::build_source("/home/users/u101373/CG-FPGA/opencl/src/FBGABlas_kernels/fblas_kernels_Direct.cl", program, context, device);
    //utils::build_source("src/kernels/kernels.cl", program , context, device);
    utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/bin/global_memory_kernels/global_memory_kernels.aocx", program, context, device);
    //utils::load_binaries("/home/users/u101373/CG-FPGA/opencl/src/FBGABlas_kernels/dgemv.aocx", program, context, device);

}

void conjugate_gradient(cl::Context &context, cl::CommandQueue &q, cl::Program &program, const double *A, const double *b, double *x, size_t n, int max_iters, double rel_error, int rank, int size) 
{

    // Marking columns for readability
    size_t m = n;

    // Split the matrix into size parts to each process
    size_t process_rows = n / size;

    // Offset indicating the starting point of the process submatrix in A
    size_t starting_offset = process_rows*m*rank;
    if(rank == size - 1)
    {
        // If it's the last process, it gets also the remaining rows
        process_rows += n % size;
    }

    // Prepare for gathering at the root
    std::vector<int> recv_counts(size); // Number of elements received from each process
    std::vector<int> displs(size);      // Displacements in the receive buffer

    if(rank == 0){
        for (int r = 0; r < size; ++r) {
            recv_counts[r] = n / size;
            if(r == size-1)
                recv_counts[r] += n % size;
            displs[r] = (r == 0) ? 0 : displs[r-1] + recv_counts[r-1];
        }
    }




    //std::cout << "Rank: " << rank << "\tStarting offset: " << starting_offset << "\tProcess rows: " << process_rows << std::endl;

    double *r = new (std::align_val_t{ 64 }) double[n];
    double *d = new (std::align_val_t{ 64 }) double[n];

    for(size_t i = 0; i < n; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        d[i] = b[i];
    }

    cl_int err;

    DirectDeviceHandler device(context, q, program);

    cl::Buffer d_A(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, process_rows*m*sizeof(double), (void*)(A+starting_offset), &err);
    CHECK_ERR(err, "d_A Buffer allocation error in CG.");
    //cl::Buffer d_b(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  size*sizeof(double), (void*)b, &err);
    //cl::Buffer d_x(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,  size*sizeof(double), (void*)x, &err);
    cl::Buffer d_d(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n*sizeof(double), (void*)d, &err);
    CHECK_ERR(err, "d_d Buffer allocation error in CG.");
   // cl::Buffer d_r(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size*sizeof(double), (void*)r, &err);

    cl::Buffer d_Ad(context, CL_MEM_READ_WRITE, m*sizeof(double));
    CHECK_ERR(err, "d_Ad Buffer allocation error in CG.");
    //cl::Buffer d_dot1(context, CL_MEM_WRITE_ONLY, 1*sizeof(double));
    // /cl::Buffer d_dot2(context, CL_MEM_WRITE_ONLY, 1*sizeof(double)); 



    double alpha, beta;
    double rr, bb;
    double dot1, dot2;


    Sequential seq;
    bb = seq.dot(b, b, n); 


    //device.dot(d_b, d_b, d_dot1, size);
    //q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &rr);
    rr = seq.dot(r, r, n); 
    bb = rr;

    int num_iters;
    

    auto start = std::chrono::high_resolution_clock::now();

    // Gemv Full result
    double * Ad = new(std::align_val_t{ 64 }) double[n];
    // Gemv partial result
    double * process_Ad = new(std::align_val_t{ 64 }) double[process_rows];

    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        //if(rank==0) 
        //    std::cout << "Iteration: " << num_iters << std::endl;
            
        device.matrix_vector_mul(d_A, d_d, d_Ad, process_rows, m);
        q.enqueueReadBuffer(d_Ad, CL_TRUE, 0, process_rows*sizeof(double), process_Ad);

        MPI_Gatherv(process_Ad, process_rows, MPI_DOUBLE, Ad, recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if(rank == 0)
        {
            
            // Calculate the dot product of Ad and d
            alpha = seq.dot(d, r, n)/(seq.dot(Ad, d, n)); 

            // Update the solution
            seq.axpby(alpha, d, 1.0, x, n);

            // Update the residual
            seq.axpby(-alpha, Ad, 1.0, r, n);

            // Calculate the beta
            beta = seq.dot(Ad, r, n)/(seq.dot(Ad, d, n)); 

            // Update the direction
            seq.axpby(1.0, r, -beta, d, n);
            rr = seq.dot(r, r, n);

        /*
        device.dot(d_d, d_r, d_dot1, size);
        device.dot(d_Ad, d_d, d_dot2, size);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);
        alpha = dot1 / dot2;

        device.vec_sum(alpha, d_d, 1.0, d_x, size);
        device.vec_sum(-alpha, d_Ad, 1.0, d_r, size);

        device.dot(d_Ad, d_r, d_dot1, size);
        device.dot(d_Ad, d_d, d_dot2, size);

        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        q.enqueueReadBuffer(d_dot2, CL_TRUE, 0, sizeof(double), &dot2);

        beta = dot1 / dot2;

        device.vec_sum(1.0, d_r, -beta, d_d, size);

        device.dot(d_r, d_r, d_dot1, size);
        q.enqueueReadBuffer(d_dot1, CL_TRUE, 0, sizeof(double), &dot1);
        rr = dot1;
        */  


        }

        MPI_Bcast(&rr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if(std::sqrt(rr / bb) < rel_error) { break; }

        MPI_Bcast(d, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        q.enqueueWriteBuffer(d_d, CL_TRUE, 0, n*sizeof(double), d);
    }

   // q.enqueueReadBuffer(d_x, CL_TRUE, 0, size*sizeof(double), x);
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] d;
    delete[] Ad;
    delete[] process_Ad;
    delete[] r;

    if(num_iters <= max_iters)
    {
       // if(rank == 0)
           // std::clog << "Rank: " << rank << "\tConverged in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
    }
    else
    {
        if(rank == 0)
            std::clog << "Rank: " << rank << "\tDid not converge in " << num_iters << " iterations, relative error is: " << std::sqrt(rr / bb) << std::endl;
    }
}

