/*****************************************************************************
 * Copyright (C) 2024 Christian Doppler Laboratory ATHENA
 *
 * Authors: Amritha Premkumar <amritha.premkumar@ieee.org>
 *          Prajit T Rajendran <prajit.rajendran@ieee.org>
 *          Vignesh V Menon <vignesh.menon@hhi.fraunhofer.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.
 *****************************************************************************/
#include <analyzer/EntropyCalculation.h>
#include <analyzer/EntropyNative.h>
#include <analyzer/common/common.h>
#include <analyzer/simd/entropy.h>

#include <cstring>

namespace vca {

double performEntropy(const unsigned blockSize,
                      const unsigned bitDepth,
                      const int16_t *pixelBuffer,
                      CpuSimd cpuSimd,
                      bool enableLowpass)
{
    std::vector<int16_t> block(blockSize * blockSize);

    // Copy pixels from pixelBuffer to block
    for (uint32_t i = 0; i < blockSize; ++i)
    {
        for (uint32_t j = 0; j < blockSize; ++j)
        {
            block[i * blockSize + j] = pixelBuffer[i * blockSize + j];
        }
    }

    double entropy    = 0;
    // Calculate entropy
    if (enableLowpass)
    {
        // Downscale the block by averaging 2x2 blocks of pixels into a single pixel
        int downscaledWidth = blockSize >> 1;
        std::vector<int16_t> downscaledBlock(downscaledWidth * downscaledWidth, 0);

        for (uint32_t i = 0; i < blockSize; i += 2)
        {
            for (uint32_t j = 0; j < blockSize; j += 2)
            {
                // Compute average pixel value of 2x2 block
                int sum = block[i * blockSize + j] + block[i * blockSize + j + 1]
                              + block[(i + 1) * blockSize + j] + block[(i + 1) * blockSize + j + 1];
                int16_t averagePixel = static_cast<int16_t>(sum >> 2);

                // Store the average pixel value in the downscaled block
                downscaledBlock[(i / 2) * downscaledWidth + (j / 2)] = averagePixel;
            }
        }
#ifdef WIN32
        if (cpuSimd == CpuSimd::AVX2)
            entropy = entropy_avx2(downscaledBlock);
        else
#endif
            entropy = vca::entropy_c(downscaledBlock);
    }
    else 
    {
#ifdef WIN32
        if (cpuSimd == CpuSimd::AVX2)
            entropy = entropy_avx2(block);
        else
#endif
            entropy = vca::entropy_c(block);
    }
    return entropy;
}

double performEdgeDensity(const unsigned blockSize,
                          const unsigned bitDepth,
                          const int16_t *pixelBuffer,
                          CpuSimd cpuSimd,
                          bool enableLowpass)
{
    // Calculate the total number of pixels in the block
    unsigned blockSizeSq = blockSize * blockSize;

    // Threshold for edge detection based on bit depth
    int threshold = (1 << (bitDepth - 1)) - 1;

    // Initialize edge count to 0
    unsigned edgeCount = 0;

    // Iterate through the pixel buffer
    for (unsigned i = 0; i < blockSizeSq; ++i)
    {
        // Check edge conditions for pixels in the buffer
        if (i % blockSize < blockSize - 1 && abs(pixelBuffer[i] - pixelBuffer[i + 1]) > threshold)
        {
            // Horizontal edge detected
            edgeCount++;
        }
        if (i / blockSize < blockSize - 1
            && abs(pixelBuffer[i] - pixelBuffer[i + blockSize]) > threshold)
        {
            // Vertical edge detected
            edgeCount++;
        }
    }

    // Calculate edge density
    double density = static_cast<double>(edgeCount) / (2 * blockSize * (blockSize - 1));

    return density;
}

} // namespace vca
