#include "utils.hpp"
#define NUM_DEVICES 1

void utils::build_source(const std::string& path, cl::Program &program, cl::Context &context) {

    std::ifstream kernel_file(path);

    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    /**
     * Compile kernel program which will run on the device.
     * */

    //cl::Program::Sources sources(1, src);
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(src.data(), src.size()));

    //Warning 
    cl::Device device = cl::Device::getDefault();
    
    program = cl::Program(context, sources);
    auto err = program.build();
    if(err != CL_BUILD_SUCCESS){
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) 
        << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }else{
        std::clog << "Program built." << std::endl;
    }
    auto binaries = program.getInfo<CL_PROGRAM_BINARIES>(&err);

     // Open the file in binary mode
    std::ofstream file("binary_output", std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    // Write the contents of the vector to the file
    file.write(reinterpret_cast<const char*>(binaries.data()), binaries.size());
    if (!file) {
        throw std::runtime_error("Failed to write to file: " + path);
    }
}

void utils::load_binaries(const std::string& path, cl::Program &program, cl::Context &context, cl::Device &device) {

    std::ifstream aocx_stream(path, std::ios::in|std::ios::binary);
    CHECK_ERR((aocx_stream.is_open() ? CL_SUCCESS:-1), "Opening .aocx");
    std::string progBuf(std::istreambuf_iterator<char>(aocx_stream),
                        (std::istreambuf_iterator<char>()));

    cl::Program::Binaries binaries;
    binaries.push_back(std::make_pair((const void*)progBuf.c_str(), progBuf.length()));

	std::vector<cl::Device> devices;
	devices.push_back(device);

/*
    std::ifstream input_file;
    input_file.open(path);
    if(!input_file.is_open()) {
        std::cout << "error in opening the file" << std::endl;
        exit(1);
    }
    size_t* length = new size_t[NUM_DEVICES];
    unsigned char* buffer;
    input_file.seekg (0, std::ios::end);
    for(int i = 0; i < NUM_DEVICES; i++) {
        length[i] = input_file.tellg();
    }
    input_file.seekg (0, std::ios::beg);
    buffer = new unsigned char [length[0]];
    input_file.read (reinterpret_cast<char *>(buffer), length[0]);
    input_file.close();
    const unsigned char** binaries = (const unsigned char**)malloc(sizeof(unsigned char*) * NUM_DEVICES);
    for(int i = 0; i < NUM_DEVICES; i++) {
        binaries[i] = buffer;
    }
    */
    std::vector<cl_int> binary_status;
    cl_int err;

    program = cl::Program(context, devices, binaries, &binary_status, &err);
    CHECK_ERR(err, "error in building the program");
}
        
bool utils::read_matrix_from_file(const char * filename, double * &matrix_out, size_t &num_rows_out, size_t &num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }
    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);

    matrix = new (std::align_val_t{ 64 }) double[num_rows * num_cols];

    fread(matrix, sizeof(double), num_rows * num_cols, file);

    matrix_out = matrix;
    num_rows_out = num_rows;
    num_cols_out = num_cols;

    fclose(file);

    return true;
}

bool utils::read_vector_from_file(const char * filename, double * &vector_out, size_t &length)
{

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&length, sizeof(size_t), 1, file);
    vector_out = new (std::align_val_t{ 64 })  double[length];

    fread(vector_out, sizeof(double), length, file);

    fclose(file);

    return true;
}

void utils::create_vector(double * &vector_out, size_t length, double scalar)
{

    vector_out = new (std::align_val_t{ 64 }) double[length];

    for(size_t i = 0; i< length; i++)
        vector_out[i] = scalar;
}

void utils::create_matrix(double * &matrix_out, size_t n, size_t m, double scalar)
{

    matrix_out = new (std::align_val_t{ 64 }) double[n*m];

    for(size_t r = 0; r<n; r++)
        for(size_t c = 0; c<m; c++)
            matrix_out[r*m + c] = scalar;
}

bool utils::read_matrix_rows(const char * filename, double * &matrix_out, size_t starting_row_num, size_t num_rows_to_read, size_t &num_cols)
{
    size_t num_rows;
    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_rows: Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    
    assert(starting_row_num + num_rows_to_read <= num_rows);

    matrix_out = new (std::align_val_t{ 64 }) double[num_rows_to_read * num_cols];

    
    size_t offset = starting_row_num * num_cols + 2; 
    if (fseek(file, sizeof(double)*offset, SEEK_SET) != 0) {
        fprintf(stderr, "read_matrix_rows: Error setting file position");
        return false;
    }

    fread(matrix_out, sizeof(double), num_rows_to_read * num_cols, file);


    fclose(file);

    return true;
}



bool utils::read_matrix_dims(const char * filename, size_t &num_rows_out, size_t &num_cols_out)
{

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "read_matrix_dims: Cannot open output file\n");
        return false;
    }

    fread(&num_rows_out, sizeof(size_t), 1, file);
    fread(&num_cols_out, sizeof(size_t), 1, file);


    fclose(file);

    return true;
}

void utils::print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file )
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}