#pragma once

#include "Matrix.h"
#include <cmath>
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
    static Matrix<uint8_t> calculateGradient(Matrix<uint8_t>& input);

private:
    uint16_t calculateGradientAtPosition(uint32_t y, uint32_t x);
    static bool Travers(Matrix<Gradient>& gradientMatrix, uint32_t x, uint32_t y);    
};

