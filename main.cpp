#include <iostream>
#include "Algorithms.h"
#include <opencv2/opencv.hpp>
#include "Matrix.h"

void showHistogram(const cv::Mat& grayscaleImage, const std::string& windowName = "Histogram") 
{
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::Mat hist;

    cv::calcHist(&grayscaleImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    int histW = 512, histH = 400;
    int binW = cvRound((double) histW / histSize);

    cv::Mat histImage(histH, histW, CV_8UC1, cv::Scalar(0));
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; ++i) 
    {
        cv::line(histImage,
            cv::Point(binW * (i - 1), histH - cvRound(hist.at<float>(i - 1))),
            cv::Point(binW * i,     histH - cvRound(hist.at<float>(i))),
            cv::Scalar(255), 2, 8, 0);
    }

    cv::imshow(windowName, histImage);
    cv::waitKey(0);
}

void showImageAndClosePrevious(const std::string& windowName, const cv::Mat& image, int delayMs = 0)
{
    cv::imshow(windowName, image);        
    if( 0 <= delayMs ) 
    {
        int key = cv::waitKey(delayMs);       
        cv::destroyWindow(windowName);        
    }
}

void showMatrixAsImage(const Matrix<uint8_t>& mat, const std::string& windowName, int delayMs = 0, bool printHistogram = false) 
{
    int rows = mat.rows();
    int cols = mat.cols();

    cv::Mat img(rows, cols, CV_8U);

    for (int y = 0; y < rows; y++) 
    {
        for (int x = 0; x < cols; x++) 
            img.at<uint8_t>(y, x) = mat.at(y, x);
    }

    if( printHistogram )
        showHistogram(img);
    
    showImageAndClosePrevious(windowName, img, delayMs);
}

void gradientRunner(Matrix<uint8_t>& input) 
{

    uint8_t lower = 13;
    for(uint16_t higher = lower; higher <= 50; higher++) 
    {
        std::cout << "Lower: " << std::to_string(lower) << " Higher: " << std::to_string(higher) << std::endl;
        Matrix<uint8_t> gradientImage = Algorithms::gradientEdgeDetection(input, lower, higher); 
        showMatrixAsImage(gradientImage, "Gradient image", 250);
    }
}

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

    for (int y = 0; y < rows; ++y) 
    {
        for (int x = 0; x < cols; ++x) 
            mat.at(y, x) = img.at<uint8_t>(y, x);
    }

    return mat;
}

int main(int argc, char *argv[]) 
{
    //const std::string imagePath = "checkerboard.png"; 
    const std::string imagePath = "Testimage.png";

    Matrix<uint8_t> input = readImageToMatrix(imagePath);

    Matrix<uint8_t> input_blurred = Algorithms::gaussian(input, 3, 6.0);
    Matrix<uint8_t> laplacianImage = Algorithms::laplacianEdgeDetection(input_blurred);
    showMatrixAsImage(laplacianImage, "Laplacian image", -1);
    
    input_blurred = Algorithms::gaussian(input, 15, 6.0);
    Matrix<uint8_t> gradientImage = Algorithms::gradientEdgeDetection(input_blurred, 4, 6); //(5,8) 
    showMatrixAsImage(gradientImage, "Gradient image", -1);

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    showImageAndClosePrevious("Original", img);

    return 0;
}
