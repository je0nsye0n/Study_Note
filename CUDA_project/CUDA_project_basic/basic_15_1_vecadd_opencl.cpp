#include <cstdio>

#include <CL/cl.h>

#include "vec_add.h"

#define CHECK_OPENCL(err)                                         \
  if (err != CL_SUCCESS) {                                        \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE);                                           \
  }

static void print_platform_info(cl_platform_id platform);
static void print_device_info(cl_device_id device);
static cl_program create_and_build_program_with_source(cl_context context,
                                                       cl_device_id device,
                                                       const char *file_name);

void vec_add_opencl(float *A, float *B, float *C, int N) {
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem a_d, b_d, c_d;

  // Get OpenCL platform
  CHECK_OPENCL(clGetPlatformIDs(1, &platform, NULL));
  print_platform_info(platform);

  // Get OpenCL device
  CHECK_OPENCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_OPENCL(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_OPENCL(err);

  // Compile program from "kernel.cl"
  program = create_and_build_program_with_source(context, device, "kernel.cl");

  // Extract kernel from compiled program
  kernel = clCreateKernel(program, "vec_add", &err);
  CHECK_OPENCL(err);

  // Create buffers
  a_d =
      clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  b_d =
      clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  c_d =
      clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);

  // Write to device
  CHECK_OPENCL(clEnqueueWriteBuffer(queue, a_d, CL_TRUE, 0, N * sizeof(float),
                                    A, 0, NULL, NULL));
  CHECK_OPENCL(clEnqueueWriteBuffer(queue, b_d, CL_TRUE, 0, N * sizeof(float),
                                    B, 0, NULL, NULL));

  // Setup kernel arguments
  CHECK_OPENCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_d));
  CHECK_OPENCL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_d));
  CHECK_OPENCL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_d));
  CHECK_OPENCL(clSetKernelArg(kernel, 3, sizeof(int), &N));

  // Setup global work size and local work size
  size_t gws[1] = {(size_t) N}, lws[1] = {256};
  for (int i = 0; i < 1; ++i) {
    // By OpenCL spec, global work size should be MULTIPLE of local work size
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  CHECK_OPENCL(
      clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL));

  // Read from device
  CHECK_OPENCL(clEnqueueReadBuffer(queue, c_d, CL_TRUE, 0, N * sizeof(float), C,
                                   0, NULL, NULL));

  CHECK_OPENCL(clReleaseMemObject(a_d));
  CHECK_OPENCL(clReleaseMemObject(b_d));
  CHECK_OPENCL(clReleaseMemObject(c_d));
  CHECK_OPENCL(clReleaseKernel(kernel));
  CHECK_OPENCL(clReleaseProgram(program));
  CHECK_OPENCL(clReleaseCommandQueue(queue));
  CHECK_OPENCL(clReleaseContext(context));
}

void print_platform_info(cl_platform_id platform) {
  size_t sz;
  char *buf;
  CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &sz));
  buf = (char *) malloc(sz);
  CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sz, buf, NULL));
  printf("Detected OpenCL platform: %s\n", buf);
  free(buf);
}

void print_device_info(cl_device_id device) {
  size_t sz;
  char *buf;
  CHECK_OPENCL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &sz));
  buf = (char *) malloc(sz);
  CHECK_OPENCL(clGetDeviceInfo(device, CL_DEVICE_NAME, sz, buf, NULL));
  printf("Detected OpenCL device: %s\n", buf);
  free(buf);
}

cl_program create_and_build_program_with_source(cl_context context,
                                                cl_device_id device,
                                                const char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    printf("Failed to open %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t source_size = ftell(file);
  rewind(file);
  char *source_code = (char *) malloc(source_size + 1);
  if (fread(source_code, sizeof(char), source_size, file) != source_size) {
    printf("Failed to read %s\n", file_name);
    exit(EXIT_FAILURE);
  }
  source_code[source_size] = '\0';
  fclose(file);
  cl_int err;
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **) &source_code, &source_size, &err);
  CHECK_OPENCL(err);
  free(source_code);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                       NULL, &log_size));
    char *log = (char *) malloc(log_size + 1);
    CHECK_OPENCL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                       log_size, log, NULL));
    log[log_size] = 0;
    printf("Compile error:\n%s\n", log);
    free(log);
  }
  CHECK_OPENCL(err);
  return program;
}