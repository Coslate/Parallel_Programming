#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include <ios>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define LOG 0
/**********************************************************************
 *	Handle errors function
 *********************************************************************/
const char *clGetErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

static void HandleError( cl_int err,
                         const char *file,
                         const int line ) {
    if(err != CL_SUCCESS){
        printf("Error(%d): %s in %s.\n", line, clGetErrorString( err ), file);
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct
{
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t align;
} RGB;

typedef struct
{
    bool type;
    uint32_t size;
    uint32_t height;
    uint32_t weight;
    RGB *data;
} Image;

Image *readbmp(const char *filename)
{
    std::ifstream bmp(filename, std::ios::binary);
    char header[54];
    bmp.read(header, 54);
    uint32_t size = *(int *)&header[2];
    uint32_t offset = *(int *)&header[10];
    uint32_t w = *(int *)&header[18];
    uint32_t h = *(int *)&header[22];
    uint16_t depth = *(uint16_t *)&header[28];
    if (depth != 24 && depth != 32)
    {
        printf("we don't suppot depth with %d\n", depth);
        exit(0);
    }
    bmp.seekg(offset, bmp.beg);

    Image *ret = new Image();
    ret->type = 1;
    ret->height = h;
    ret->weight = w;
    ret->size = w * h;
    ret->data = new RGB[w * h];

    uint32_t residue   = ret->size%32;
    for (int k = 0; k < residue; ++k){
        bmp.read((char *)&ret->data[k], depth / 8);
    }

    for (int i = residue; i < ret->size; i+=32)
    {
        bmp.read((char *)&ret->data[i], depth / 8);
        bmp.read((char *)&ret->data[i+1], depth / 8);
        bmp.read((char *)&ret->data[i+2], depth / 8);
        bmp.read((char *)&ret->data[i+3], depth / 8);
        bmp.read((char *)&ret->data[i+4], depth / 8);
        bmp.read((char *)&ret->data[i+5], depth / 8);
        bmp.read((char *)&ret->data[i+6], depth / 8);
        bmp.read((char *)&ret->data[i+7], depth / 8);
        bmp.read((char *)&ret->data[i+8], depth / 8);
        bmp.read((char *)&ret->data[i+9], depth / 8);
        bmp.read((char *)&ret->data[i+10], depth / 8);
        bmp.read((char *)&ret->data[i+11], depth / 8);
        bmp.read((char *)&ret->data[i+12], depth / 8);
        bmp.read((char *)&ret->data[i+13], depth / 8);
        bmp.read((char *)&ret->data[i+14], depth / 8);
        bmp.read((char *)&ret->data[i+15], depth / 8);
        bmp.read((char *)&ret->data[i+16], depth / 8);
        bmp.read((char *)&ret->data[i+17], depth / 8);
        bmp.read((char *)&ret->data[i+18], depth / 8);
        bmp.read((char *)&ret->data[i+19], depth / 8);
        bmp.read((char *)&ret->data[i+20], depth / 8);
        bmp.read((char *)&ret->data[i+21], depth / 8);
        bmp.read((char *)&ret->data[i+22], depth / 8);
        bmp.read((char *)&ret->data[i+23], depth / 8);
        bmp.read((char *)&ret->data[i+24], depth / 8);
        bmp.read((char *)&ret->data[i+25], depth / 8);
        bmp.read((char *)&ret->data[i+26], depth / 8);
        bmp.read((char *)&ret->data[i+27], depth / 8);
        bmp.read((char *)&ret->data[i+28], depth / 8);
        bmp.read((char *)&ret->data[i+29], depth / 8);
        bmp.read((char *)&ret->data[i+30], depth / 8);
        bmp.read((char *)&ret->data[i+31], depth / 8);
    }
    return ret;
}

int writebmp(const char *filename, Image *img)
{

    uint8_t header[54] = {
        0x42,        // identity : B
        0x4d,        // identity : M
        0, 0, 0, 0,  // file size
        0, 0,        // reserved1
        0, 0,        // reserved2
        54, 0, 0, 0, // RGB data offset
        40, 0, 0, 0, // struct BITMAPINFOHEADER size
        0, 0, 0, 0,  // bmp width
        0, 0, 0, 0,  // bmp height
        1, 0,        // planes
        32, 0,       // bit per pixel
        0, 0, 0, 0,  // compression
        0, 0, 0, 0,  // data size
        0, 0, 0, 0,  // h resolution
        0, 0, 0, 0,  // v resolution
        0, 0, 0, 0,  // used colors
        0, 0, 0, 0   // important colors
    };

    // file size
    uint32_t file_size = img->size * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    // width
    uint32_t width = img->weight;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    // height
    uint32_t height = img->height;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream fout;
    fout.open(filename, std::ios::binary);
    fout.write((char *)header, 54);
    fout.write((char *)img->data, img->size * 4);
    fout.close();
}

inline void LoadProgram(cl_context context, const char *file_name, cl_program &program) {
    FILE* program_handle;
    size_t program_size;
    char *program_buffer;
    cl_int ret_code = 0;

    // get size of kernel source
    program_handle = fopen(file_name, "r");
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);

    // read kernel source into buffer
    program_buffer = (char*) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    //printf("source code = \n %s\n", program_buffer);

    // create program from buffer
    program = clCreateProgramWithSource(context, 1, (const char**) &program_buffer, &program_size, &ret_code);
    free(program_buffer);
}

int main(int argc, char *argv[]){
    cl_uint             plat_num        = 0;
    cl_platform_id      plat_id         = 0;
    cl_device_id        device_id       = 0;
	cl_uint             device_num      = 0;
    cl_int              ret_code        = 0;
    size_t              device_num_b    = 0;
    size_t              value_b         = 0;
    char*               device_name     = NULL;
    char*               device_version  = NULL;
    char*               driver_version  = NULL;
    char*               opencl_version  = NULL;
    cl_uint             maxComputeUnits = 0;
    cl_uint             maxWorkItemDimension = 0;
    size_t             *maxWorkItemSize = NULL;
    size_t              maxWorkGroupSize = 0;
    cl_context          context;
    cl_program          kernel_program;


    //-------------------Get platform & device---------------------//
    clGetPlatformIDs(1, &plat_id, &plat_num);
    clGetDeviceIDs(plat_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &device_num);

    //-------------------Create context-----------------------------//
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret_code);

    //-------------------Create a command queue---------------------//
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret_code);

    //-------------------Build kernel------------------------------//
    LoadProgram(context, "./histogram.cl", kernel_program);

#if LOG
    time_t t;
	srand((unsigned) time(&t));
	int some_rand_num = rand() % 100 + 200; // 200 to 299
    const unsigned MAX_INFO_SIZE = 0x10000;
    char options[MAX_INFO_SIZE];
	sprintf(options, "-cl-nv-maxrregcount=%d -cl-nv-verbose", some_rand_num); // randomn number is added to avoid empty log
    size_t log_size;
    char *log;

    cl_int err_t = clBuildProgram(kernel_program, 1, &device_id, options, NULL, NULL);
    clGetProgramBuildInfo(kernel_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    log = (char*) malloc(log_size+1);
    clGetProgramBuildInfo(kernel_program, device_id, CL_PROGRAM_BUILD_LOG, MAX_INFO_SIZE, log, NULL);
    log[log_size] = '\0';

    std::cout<<"log = "<<log<<std::endl;
    std::cout<<"err_t = "<<err_t<<std::endl;
#else
    cl_int err_t = clBuildProgram(kernel_program, 1, &device_id, NULL, NULL, NULL);
#endif

    cl_kernel kernel_obj     = clCreateKernel(kernel_program, "histogram", &ret_code);
    cl_kernel kernel_ret_obj = clCreateKernel(kernel_program, "present_result", &ret_code);
    cl_kernel kernel_max_obj = clCreateKernel(kernel_program, "find_hist_max", &ret_code);


     //------------------Processing multiple images------------------//
    char *filename;
    if (argc >= 2)
    {
        int many_img = argc - 1;

        for (int i = 0; i < many_img; i++)
        {
            filename = argv[i + 1];
            Image *img = readbmp(filename);

            std::cout << img->weight << ":" << img->height << "\n";

            //------------------Kernel work group/item setting------------------//
            size_t local_work_size[2] = {32, 32};
            int num_groups_x = (img->weight+local_work_size[0]-1)/local_work_size[0];
            int num_groups_y = (img->height+local_work_size[1]-1)/local_work_size[1];
            int total_groups = num_groups_x * num_groups_y;
            size_t global_work_size[2] = {num_groups_x * local_work_size[0], num_groups_y * local_work_size[1]};

            size_t local_work_size_max[1] = {256};
            size_t global_work_size_max[1] = {768};
            
            //------------------Memory allocation on host------------------//
            uint32_t *hist_calc_h = (uint32_t*) malloc (sizeof(uint32_t) * 769);//the last one is for max value storage
            memset(hist_calc_h, 0, sizeof(uint32_t) * 769);

            //------------------Memory allocation on device------------------//
            cl_mem orig_img_d      = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * 4 * img->size, NULL, &ret_code);
            cl_mem ret_img_d       = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * 262144, NULL, &ret_code);
            cl_mem hist_calc_d     = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t) * 769, NULL, &ret_code);
            
            //------------------Memory host to device------------------//
            clEnqueueWriteBuffer(command_queue, orig_img_d, CL_TRUE, 0, sizeof(uint8_t) * 4 * img->size, img->data, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, hist_calc_d, CL_TRUE, 0, sizeof(uint32_t) * 769, hist_calc_h, 0, NULL, NULL);


            //-------------------Set kernel arguments-----------------//
            clSetKernelArg(kernel_obj, 0, sizeof(cl_mem), &orig_img_d);
            clSetKernelArg(kernel_obj, 1, sizeof(cl_mem), &hist_calc_d);
            clSetKernelArg(kernel_obj, 2, sizeof(uint32_t), &img->height);
            clSetKernelArg(kernel_obj, 3, sizeof(uint32_t), &img->weight);

            clSetKernelArg(kernel_max_obj, 0, sizeof(cl_mem), &hist_calc_d);
            //-------------------Execute kernel-histogram function--------------//
            clEnqueueNDRangeKernel(command_queue, kernel_obj, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

            //-------------------Execute kernel-find_hist_max function--------------//
            clEnqueueNDRangeKernel(command_queue, kernel_max_obj, 1, NULL, global_work_size_max, local_work_size_max, 0, NULL, NULL);
            
            //-------------------Read the result back to host--------//
            Image *ret = new Image();
            ret->type = 1;
            ret->height = 256;
            ret->weight = 256;
            ret->size = 65536;
            ret->data = new RGB[65536]{};

            //-------------------Set kernel arguments-----------------//
            clSetKernelArg(kernel_ret_obj, 0, sizeof(cl_mem), &ret_img_d);
            clSetKernelArg(kernel_ret_obj, 1, sizeof(cl_mem), &hist_calc_d);

            //-------------------Execute kernel function--------------//
            local_work_size[0] = 32;
            local_work_size[1] = 32;
            global_work_size[0] = 256;
            global_work_size[1] = 256;

            clEnqueueNDRangeKernel(command_queue, kernel_ret_obj, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

            //-------------------Read the result back to host--------//
            clEnqueueReadBuffer(command_queue, ret_img_d, CL_TRUE, 0, sizeof(uint8_t) * 262144, ret->data, 0, NULL, NULL);

            std::string newfile = "hist_" + std::string(filename); 
            writebmp(newfile.c_str(), ret);

            free(hist_calc_h);
            clReleaseMemObject(hist_calc_d);
            clReleaseMemObject(orig_img_d);
            clReleaseMemObject(ret_img_d);
        }
    }else{
        printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
    }

    /*
    clReleaseKernel(kernel_obj);
    clReleaseKernel(kernel_ret_obj);
    clReleaseKernel(kernel_max_obj);
    clReleaseProgram(kernel_program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    */
    return EXIT_SUCCESS;
}
