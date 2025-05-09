#pragma once

#include <iostream>
#include <vector>
#include "Pixel.h"
#include <omp.h>

template <typename T>
class Matrix {
private:
    uint32_t m_Height;
    uint32_t m_Width;
    std::vector<Pixel<T>> m_Matrix;

public:
    using MatrixCallback = void (*)(Pixel<T>&, uint32_t, uint32_t);

    Matrix(int h, int w) : m_Height(h), m_Width(w), m_Matrix(h * w) {}

    Pixel<T>& at(int y, int x) { return m_Matrix[y * m_Width + x]; }
    const Pixel<T>& at(int y, int x) const { return m_Matrix[y * m_Width + x]; }

    int rows() const { return m_Height; }
    int cols() const { return m_Width; }

    Pixel<T>* data() { return m_Matrix.data(); }
    const Pixel<T>* data() const { return m_Matrix.data(); }

    auto begin() { return m_Matrix.begin(); }
    auto end() { return m_Matrix.end(); }
    auto begin() const { return m_Matrix.begin(); }
    auto end() const { return m_Matrix.end(); }

    void iterate(MatrixCallback cb) {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < m_Height; ++y) {
            for (int x = 0; x < m_Width; ++x) {
                cb(at(y, x), y, x);
            }
        }
    }
};
