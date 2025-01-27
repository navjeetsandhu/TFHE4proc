// Author: Arjun Ramaswami

#ifndef KERNEL_VARS
#define KERNEL_VARS

#include "CL/opencl.h"

extern cl_platform_id platform;
extern cl_device_id *devices;
extern cl_device_id device;
extern cl_context context;
extern cl_program program;
extern cl_command_queue queue1, queue2, queue3;
extern cl_command_queue queue4, queue5, queue6;
extern cl_command_queue queue7, queue8;

extern bool svm_enabled;

extern void queue_setup_1();
extern void queue_cleanup_1();

extern void queue_setup_2();
extern void queue_cleanup_2();

extern void queue_setup_3();
extern void queue_cleanup_3();

extern void queue_setup_4();
extern void queue_cleanup_4();

#endif
