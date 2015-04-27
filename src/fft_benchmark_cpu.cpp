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

#include <fftw3.h>

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

    for (int n = NSamplesMin; n <= NSamplesMax; n <<= 1) {
        int fft_dims[3];
        int batch_dist = 1;
        for (int i = 0; i < Rank; i++) {
            fft_dims[i] = n;
            batch_dist *= fft_dims[i];
        }

        size_t bufferLen = batch_dist * 2 * Batches;

        // No data is copied to buffers as FFT performance is not data
        // dependent, but only size dependent
        float* bufferIn  = new float[bufferLen];
        float* bufferOut = new float[bufferLen];

        std::cout << "Number of dimensions: " << Rank << std::endl;
        std::cout << "Matrix dimensions: " << n << "x" << n << std::endl;
        std::cout << "Batch size: " << Batches << std::endl;

        // In-place plan
        fftwf_plan inPlan = fftwf_plan_many_dft(Rank, fft_dims, Batches,
                                                (fftwf_complex*)bufferOut, NULL, 1, batch_dist,
                                                (fftwf_complex*)bufferOut, NULL, 1, batch_dist,
                                                FFTW_FORWARD, FFTW_ESTIMATE);

        // Make sure the plan is properly initialized before benchmarking
        fftwf_execute(inPlan);

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // Compute forward FFT
            fftwf_execute(inPlan);
        }
        tEnd = getTime();
        std::cout << "In-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        // Destroy in-place plan
        fftwf_destroy_plan(inPlan);

        // Out-of-place plan
        fftwf_plan outPlan = fftwf_plan_many_dft(Rank, fft_dims, Batches,
                                                 (fftwf_complex*)bufferIn, NULL, 1, batch_dist,
                                                 (fftwf_complex*)bufferOut, NULL, 1, batch_dist,
                                                 FFTW_FORWARD, FFTW_ESTIMATE);

        // Make sure the plan is properly initialized before benchmarking
        fftwf_execute(outPlan);

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            // Compute forward FFT
            fftwf_execute(outPlan);
        }
        tEnd = getTime();
        std::cout << "Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl;

        tStart = getTime();
        for (int i = 0; i < Iterations; i++) {
            memcpy(bufferOut, bufferIn, bufferLen * sizeof(float));
            //for (int j = 0; j < bufferLen; j++)
            //    bufferOut[j] = bufferIn[j];
            // Compute forward FFT
            fftwf_execute(outPlan);
        }
        tEnd = getTime();
        std::cout << "Buffer Copy + Out-of-place C2C FFT time for " << Iterations << " runs: " << getTimeCount(tEnd, tStart) << " ms" << std::endl << std::endl;

        // Destroy out-of-place plan
        fftwf_destroy_plan(outPlan);

        delete[] bufferIn;
        delete[] bufferOut;
    }

    return 0;
}
