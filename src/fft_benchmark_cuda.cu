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

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        cudaDeviceReset();
        exit(-1);
    }
}

#define CUDA_CHECK(val)           check ( (val), #val, __FILE__, __LINE__ )

static const int NSamplesMin = 128;
static const int NSamplesMax = 4096;
static const int Batches = 10;
static const int Rank = 2;
static const int Iterations = 10;

int main()
{
#if __cplusplus > 199711L
        std::chrono::high_resolution_clock::time_point tStart, tEnd;
#else
        std::clock_t tStart, tEnd;
#endif

    for (int n = NSamplesMin; n <= NSamplesMax; n <<= 1) {
        cufftHandle plan_c2c;
        cufftComplex *d_complex, *d_c2c_out;

        size_t in_sz = n;
        for (int i = 1; i < Rank; i++)
            in_sz *= n;

        int dims[] = {n, n, n};

        size_t bufferSize = in_sz * Batches * sizeof(cufftComplex);

        // No data is copied to buffers as FFT performance is not data
        // dependent, but only size dependent
        CUDA_CHECK(cudaMalloc((void**)&d_complex, bufferSize));
        CUDA_CHECK(cudaMalloc((void**)&d_c2c_out, bufferSize));

        // Allocate cuFFT plan
        CUDA_CHECK(cufftPlanMany(&plan_c2c, Rank, dims, NULL, 0, 0,
                                 NULL, 0, 0, CUFFT_C2C, Batches));

        std::cout << "Number of dimensions: " << Rank << std::endl;
        std::cout << "Matrix dimensions: " << n << "x" << n << std::endl;
        std::cout << "Batch size: " << Batches << std::endl << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // In-place plan
            CUDA_CHECK(cufftExecC2C(plan_c2c, d_c2c_out, d_c2c_out, CUFFT_FORWARD));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        tEnd = getTime();
        std::cout << "In-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // Out-of-place plan
            CUDA_CHECK(cufftExecC2C(plan_c2c, d_complex, d_c2c_out, CUFFT_FORWARD));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        tEnd = getTime();
        std::cout << "Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            CUDA_CHECK(cudaMemcpy(d_complex, d_c2c_out, bufferSize, cudaMemcpyDeviceToDevice));
            // Out-of-place plan
            CUDA_CHECK(cufftExecC2C(plan_c2c, d_complex, d_c2c_out, CUFFT_FORWARD));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        tEnd = getTime();
        std::cout << "Buffer Copy + Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl << std::endl;

        // Destroy plan
        CUDA_CHECK(cufftDestroy(plan_c2c));

        // Free CUDA buffers
        CUDA_CHECK(cudaFree(d_complex));
        CUDA_CHECK(cudaFree(d_c2c_out));
    }
}
