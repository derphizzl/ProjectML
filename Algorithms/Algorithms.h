#pragma once

#include "Matrix.h"
#include <cmath>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

struct Gradient {
    double dx;
    double dy;
    double val;
    int deg;
    int dirX;
    int dirY;
    bool visited;
    uint8_t binary;
    
    struct {
        double lowerThreshold;
        double higherThreshold;
    } Threshold;

    Gradient() { 
        visited = false; 
        Threshold.lowerThreshold = .0; 
        Threshold.higherThreshold = .0; 
    }
};


class Algorithms {
public:
    static double toDegrees(const double& rad) { return rad * 180.0/M_PI; }
    static double toRadians(const double& degrees) { return degrees * M_PI/180.0; }
    static Matrix<uint8_t> gradientEdgeDetection(Matrix<uint8_t>& input, const uint8_t& lower_threshold, const uint8_t& higher_threshold);
    static Matrix<uint8_t> laplacianEdgeDetection(Matrix<uint8_t>& input);
    static Matrix<uint8_t> gaussian(Matrix<uint8_t>& input, const uint8_t& kernel_size, const double& sigma_factor=6.0);

private:
    uint16_t calculateGradientAtPosition(uint32_t y, uint32_t x);
    static bool Travers(Matrix<Gradient>& gradientMatrix, uint32_t x, uint32_t y);   
    static FilterKernel gaussianKernel(const uint8_t& kernel_size, const double& sigma);
};

