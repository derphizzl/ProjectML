#include <iostream>
#include "Algorithms/Algorithms.h"

#include <opencv2/opencv.hpp>
#include "Primitive/Matrix.h"  // your custom Matrix<T> class

Matrix<uint8_t> readImageToMatrix(const std::string& path) 
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (img.empty()) 
        throw std::runtime_error("Failed to load image: " + path);

    if( img.depth() != CV_8U ) 
        img.convertTo(img, CV_8U);

    int rows = img.rows;
    int cols = img.cols;

    Matrix<uint8_t> mat(rows, cols);

    // 3. Copy pixel values
    for (int y = 0; y < rows; ++y) 
    {
        for (int x = 0; x < cols; ++x) 
            mat.at(y, x) = img.at<uint8_t>(y, x);
    }

    return mat;
}

void showMatrixAsImage(const Matrix<uint8_t>& mat, const std::string& windowName) 
{
    int rows = mat.rows();
    int cols = mat.cols();

    cv::Mat img(rows, cols, CV_8U);

    for (int y = 0; y < rows; ++y) 
    {
        for (int x = 0; x < cols; ++x) 
            img.at<uint8_t>(y, x) = mat.at(y, x);
    }

    cv::imshow(windowName, img);
}

int main(int argc, char *argv[]) 
{
    const std::string imagePath = "/data/local_workspace/007_CV/01_Source/ImageProcessingNoQt/Testimage.png";
    
    Matrix<uint8_t> input = readImageToMatrix(imagePath);
    Matrix<uint8_t> gradientImage = Algorithms::calculateGradient(input);
    
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    cv::imshow("Original", img);
    
    showMatrixAsImage(gradientImage, "Gradient image");
    
    cv::waitKey(0);
    return 0;
}
