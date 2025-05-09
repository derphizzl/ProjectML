#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>

template <typename T> class Pixel {
private:
    uint32_t m_X;
    uint32_t m_Y;
    T m_R;
    T m_G;
    T m_B;
    T m_A;
    T m_GS;
public:
    Pixel(const T& red, const T& green, const T& blue, const T& alpha=T(), const uint32_t& x=0, const uint32_t& y=0) : 
        m_X(x), m_Y(y), m_R(red), m_G(green), m_B(blue),m_A(alpha), m_GS(0) {}

    Pixel(const T& grey, const uint32_t& x, const uint32_t& y) :
        m_X(x), m_Y(y), m_R(T()), m_G(T()), m_B(T()), m_A(T()), m_GS(grey) {}

    Pixel(const T& grey) :
        m_X(0), m_Y(0), m_R(T()), m_G(T()), m_B(T()), m_A(T()), m_GS(grey) {}
    
    void setRed(const T& red) { m_R = red; }
    T red() const { return m_R; }
    
    void setGreen(const T& green) { m_G = green; }
    T green() const { return m_G; }
    
    void setBlue(const T& blue) { m_B = blue; }
    T blue() const { return m_B; }
    
    void setAlpha(const T& alpha) { m_A = alpha; }
    T alpha() const { return m_A; }
    
    void setGrey(const T& grey) { m_GS = grey; }
    T grey() const { return m_GS; }

    void setX(const uint32_t& x) { m_X = x; }
    uint32_t x() const { return m_X; }
    
    void setY(const uint32_t& y) { m_Y = y; }
    uint32_t y() const { return m_Y; }
};
    
