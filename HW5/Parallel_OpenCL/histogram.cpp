#include <iostream>
#include <cstdio>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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


    //-------------------Get platform & device---------------------//
    HANDLE_ERROR(clGetPlatformIDs(1, &plat_id, &plat_num));
    HANDLE_ERROR(clGetDeviceIDs(plat_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &device_num));
    
    /*
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &device_num_b));
    device_name = (char*) malloc(device_num_b);
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_num_b, device_name, &device_num_b));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &value_b));
    device_version = (char*) malloc(value_b);
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_VERSION, value_b, device_version, NULL));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &value_b));
    driver_version = (char*) malloc(value_b);
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DRIVER_VERSION, value_b, driver_version, NULL));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &value_b));
    opencl_version = (char*) malloc(value_b);
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, value_b, opencl_version, NULL));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimension), &maxWorkItemDimension, NULL));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL));

    maxWorkItemSize = new size_t[maxWorkItemDimension]();
    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimension*sizeof(size_t), maxWorkItemSize, NULL));

    HANDLE_ERROR(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL));
    */

    /*
    std::cout<<"plat_num         = "<<plat_num<<std::endl;
    std::cout<<"plat_id          = "<<plat_id<<std::endl;
    std::cout<<"device_num       = "<<device_num<<std::endl;
    std::cout<<"device_num_b     = "<<device_num_b<<std::endl;
    std::cout<<"device_id        = "<<device_id<<std::endl;
    std::cout<<"device_name      = "<<device_name<<std::endl;
    std::cout<<"device_version   = "<<device_version<<std::endl;
    std::cout<<"driver_version   = "<<driver_version<<std::endl;
    std::cout<<"opencl_version   = "<<opencl_version<<std::endl;
    std::cout<<"max_compite_units= "<<maxComputeUnits<<std::endl;
    std::cout<<"max_work_item_dimension = "<<maxWorkItemDimension<<std::endl;
    for(int i=0;i<3;++i){
        std::cout<<"max_work_item_size["<<i<<"] = "<<maxWorkItemSize[i]<<std::endl;
    }
    std::cout<<"max_work_group_size = "<<maxWorkGroupSize<<std::endl;
    */
    //-------------------Create context---------------------//
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret_code);
    HANDLE_ERROR(ret_code);

    //-------------------Create a command queue---------------------//
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret_code);
    HANDLE_ERROR(ret_code);

    clReleaseContext(context);
    return EXIT_SUCCESS;
}
