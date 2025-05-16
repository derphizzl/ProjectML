#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <omp.h>
#include <algorithm>

using FilterKernel = std::vector<std::vector<double>>;

template <typename T>
class Matrix {
private:
    uint32_t m_Height;
    uint32_t m_Width;
    std::vector<T> m_Matrix;

public:
    Matrix(uint32_t h, uint32_t w) : m_Height(h), m_Width(w), m_Matrix(h * w) {}

    T& at(uint32_t y, uint32_t x) { return m_Matrix[y * m_Width + x]; }
    const T& at(uint32_t y, uint32_t x) const { return m_Matrix[y * m_Width + x]; }

    uint32_t rows() const { return m_Height; }
    uint32_t cols() const { return m_Width; }

    T* data() { return m_Matrix.data(); }
    const T* data() const { return m_Matrix.data(); }

    auto begin() { return m_Matrix.begin(); }
    auto end() { return m_Matrix.end(); }
    auto begin() const { return m_Matrix.begin(); }
    auto end() const { return m_Matrix.end(); }
    
    Matrix<T>& operator=(const Matrix<T>& other) 
    {
        if( this != &other ) 
        {
            m_Height = other.m_Height;
            m_Width = other.m_Width;
            m_Matrix = other.m_Matrix;
        }
        
        return *this;
    }
    
    template <typename Func> void iterate(Func&& cb) 
    {
       #pragma omp parallel for collapse(2)
        for (uint32_t y = 0; y < m_Height; ++y) 
        {
            for (uint32_t x = 0; x < m_Width; ++x) 
            {
                cb(at(y, x), y, x);
            }
        }
    }

    template <typename Func> void iterateRegion(uint32_t yStart, uint32_t yEnd, uint32_t xStart, uint32_t xEnd, Func&& cb) 
    {
        #pragma omp parallel for collapse(2)
        for (uint32_t y = yStart; y < static_cast<uint32_t>(yEnd); ++y) 
        {
            for (uint32_t x = xStart; x < static_cast<uint32_t>(xEnd); ++x) 
            {
                cb(at(y, x), y, x);
            }
        }
    }

    // Matrix<T> convolve(const FilterKernel& kernel) 
    // {
    //     int kernelSize = kernel.size();
    //     int kHalf = kernelSize / 2;
    //
    //     Matrix<T> output(rows(), cols());
    //
    //     output.iterate([&](T& outVal, uint32_t y, uint32_t x) 
    //     {
    //         if (y < kHalf || y >= rows() - kHalf ||
    //             x < kHalf || x >= cols() - kHalf)
    //         {
    //             outVal = 0;
    //             return;
    //         }
    //
    //         float sum = 0.0f;
    //
    //         for (int ky = 0; ky < kernelSize; ++ky) 
    //         {
    //             for (int kx = 0; kx < kernelSize; ++kx) 
    //             {
    //                 int iy = y + ky - kHalf;
    //                 int ix = x + kx - kHalf;
    //                 sum += at(iy, ix) * kernel[ky][kx];
    //             }
    //         }
    //
    //         outVal = static_cast<T>(std::clamp(sum, 0.0f, 255.0f));
    //     });
    //
    //     return output;
    // }

    Matrix<T> convolve(const FilterKernel& kernel) 
    {
        int kernelSize = kernel.size();
        int kHalf = kernelSize / 2;

        Matrix<T> output(rows(), cols());

        output.iterate([&](T& outVal, uint32_t y, uint32_t x) 
        {
            float sum = 0.0f;

            for (int ky = 0; ky < kernelSize; ++ky) 
            {
                for (int kx = 0; kx < kernelSize; ++kx) 
                {
                    int iy = std::clamp<int>(y + ky - kHalf, 0, rows() - 1);
                    int ix = std::clamp<int>(x + kx - kHalf, 0, cols() - 1);

                    sum += at(iy, ix) * kernel[ky][kx];
                }
            }

            outVal = static_cast<T>(std::clamp(sum, 0.0f, 255.0f));
        });

        return output;
    }

};
