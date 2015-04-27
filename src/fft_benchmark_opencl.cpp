/*********************************************************************************
 * Copyright (c) 2015, Peter Andreas Entschev
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of fft_benchmark nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

#include "common.hpp"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <clFFT.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CLFFT_CHECK(err) {                  \
    if (err != CLFFT_SUCCESS) {             \
        printf("Error %d in %s:%d.\n",      \
               err, __FILE__, __LINE__);    \
        return -1;                          \
    }                                       \
}

static const int NSamplesMin = 128;
static const int NSamplesMax = 4096;
static const int Batches = 10;
static const int Rank = 2;
static const int Iterations = 10;

int main(int argc, char* argv[])
{
#if __cplusplus > 199711L
    std::chrono::high_resolution_clock::time_point tStart, tEnd;
#else
    std::clock_t tStart, tEnd;
#endif

    // Query platforms and devices
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_devices, num_platforms;
    cl_int err;
    CLFFT_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));
    CLFFT_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1,
                         &device, &num_devices));

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CLFFT_CHECK(err);

    // Create OpenCL command queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CLFFT_CHECK(err);

    for (int n = NSamplesMin; n <= NSamplesMax; n <<= 1) {
        clfftPlanHandle inPlan, outPlan;
        clfftDim rank = (clfftDim)Rank;
        size_t fft_strides[3], fft_dims[3];
        for (int i = 0; i < Rank; i++) {
            fft_dims[i] = n;
            fft_strides[i] = (i == 0) ? 1 : fft_strides[i-1] * fft_dims[i];
        }

        // Setup clFFT
        clfftSetupData fftSetup;
        CLFFT_CHECK(clfftInitSetupData(&fftSetup));
        CLFFT_CHECK(clfftSetup(&fftSetup));

        // Create a complex clFFT plan
        CLFFT_CHECK(clfftCreateDefaultPlan(&inPlan, context, rank, fft_dims));

        // Configure clFFT inPlan
        CLFFT_CHECK(clfftSetLayout(inPlan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
        CLFFT_CHECK(clfftSetPlanBatchSize(inPlan, Batches));
        CLFFT_CHECK(clfftSetPlanDistance(inPlan, fft_strides[Rank], fft_strides[Rank]));
        CLFFT_CHECK(clfftSetPlanInStride(inPlan, rank, fft_strides));
        CLFFT_CHECK(clfftSetPlanOutStride(inPlan, rank, fft_strides));
        CLFFT_CHECK(clfftSetPlanPrecision(inPlan, CLFFT_SINGLE));
        CLFFT_CHECK(clfftSetResultLocation(inPlan, CLFFT_INPLACE));

        // Bake inPlan
        CLFFT_CHECK(clfftBakePlan(inPlan, 1, &queue, NULL, NULL));

        cl_mem complexIn = 0, complexOut = 0;
        size_t bufferSize = sizeof(float) * 2 * Batches;
        for (int i = 0; i < Rank; i++)
            bufferSize *= n;

        // No data is copied to buffers as FFT performance is not data
        // dependent, but only size dependent
        complexIn  = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &err);
        CLFFT_CHECK(err);
        complexOut = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize, 0, &err);
        CLFFT_CHECK(err);

        std::cout << "Matrix dimensions: " << n << "x" << n << std::endl;
        std::cout << "Batch size: " << Batches << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // Compute forward FFT
            CLFFT_CHECK(clfftEnqueueTransform(inPlan, CLFFT_FORWARD, 1, &queue,
                                              0, NULL, NULL,
                                              &complexOut, &complexOut, NULL));
        }
        CLFFT_CHECK(clFinish(queue));
        tEnd = getTime();
        std::cout << "In-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        // Destroy in-place FFT plan
        CLFFT_CHECK(clfftDestroyPlan(&inPlan));

        // Create a complex clFFT plan
        CLFFT_CHECK(clfftCreateDefaultPlan(&outPlan, context, rank, fft_dims));

        // Configure clFFT outPlan
        CLFFT_CHECK(clfftSetLayout(outPlan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
        CLFFT_CHECK(clfftSetPlanBatchSize(outPlan, Batches));
        CLFFT_CHECK(clfftSetPlanDistance(outPlan, fft_strides[Rank], fft_strides[Rank]));
        CLFFT_CHECK(clfftSetPlanInStride(outPlan, rank, fft_strides));
        CLFFT_CHECK(clfftSetPlanOutStride(outPlan, rank, fft_strides));
        CLFFT_CHECK(clfftSetPlanPrecision(outPlan, CLFFT_SINGLE));
        CLFFT_CHECK(clfftSetResultLocation(outPlan, CLFFT_INPLACE));

        // Bake outPlan
        CLFFT_CHECK(clfftBakePlan(outPlan, 1, &queue, NULL, NULL));

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // Compute forward FFT
            CLFFT_CHECK(clfftEnqueueTransform(outPlan, CLFFT_FORWARD, 1, &queue,
                                              0, NULL, NULL,
                                              &complexIn, &complexOut, NULL));
        }
        CLFFT_CHECK(clFinish(queue));
        tEnd = getTime();
        std::cout << "Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            CLFFT_CHECK(clEnqueueCopyBuffer(queue, complexOut, complexIn, 0, 0, bufferSize, 0, NULL, NULL));
            // Compute forward FFT
            CLFFT_CHECK(clfftEnqueueTransform(outPlan, CLFFT_FORWARD, 1, &queue,
                                              0, NULL, NULL,
                                              &complexIn, &complexOut, NULL));
        }
        CLFFT_CHECK(clFinish(queue));
        tEnd = getTime();
        std::cout << "Buffer Copy + Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl << std::endl;

        CLFFT_CHECK(clReleaseMemObject(complexIn));
        CLFFT_CHECK(clReleaseMemObject(complexOut));

        // Destroy out-of-place FFT plan
        CLFFT_CHECK(clfftDestroyPlan(&outPlan));

        // Force release of clFFT temporary buffers
        CLFFT_CHECK(clfftTeardown());
    }

    // Flush/Release OpenCL resources
    CLFFT_CHECK(clFlush(queue));
    CLFFT_CHECK(clFinish(queue));
    CLFFT_CHECK(clReleaseCommandQueue(queue));
    CLFFT_CHECK(clReleaseContext(context));

    return 0;
}
