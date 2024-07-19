#ifndef OPENCL_UTILS_HPP
#define OPENCL_UTILS_HPP

#define CHECK_ERR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

#endif
