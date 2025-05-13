#include "Algorithms.h"
#include "Logging.h"
#include "Matrix.h"
#include <cwchar>

bool Algorithms::Travers(Matrix<Gradient>& gradientMatrix, uint32_t y, uint32_t x)
{
	// Bounds check
	if (x <= 0 || x >= gradientMatrix.cols() - 1 ||
	    y <= 0 || y >= gradientMatrix.rows() - 1)
		return false;

	if( gradientMatrix.at(y, x).visited )
		return true;

	constexpr float DIR_TOLERANCE = 1.0f;//0.1f;

	// Normalize helper
	auto normalize = [](float dx, float dy) -> std::pair<float, float> {
		float len = std::sqrt(dx * dx + dy * dy);
		if (len == 0.0f) return {0.0f, 0.0f};
		return {dx / len, dy / len};
	};  

	// Current direction
	auto& center = gradientMatrix.at(y, x);
	auto [dirX, dirY] = normalize(center.dirX, center.dirY);

	// 8-connected neighbors
	static const int dx[8] = { 1,  0, -1,  0, -1,  1, -1,  1 };
	static const int dy[8] = { 0,  1,  0, -1, -1, -1,  1,  1 };

	for (int i = 0; i < 8; ++i) 
	{
		int nx = x + dx[i];
		int ny = y + dy[i];

		// Neighbor bounds safety
		if (nx < 0 || nx >= gradientMatrix.cols() ||
		    ny < 0 || ny >= gradientMatrix.rows() )
			continue;

		auto& neighbor = gradientMatrix.at(ny, nx);
		auto [nDirX, nDirY] = normalize(neighbor.dirX, neighbor.dirY);

		bool similarDirection =
			std::abs(nDirX - dirX) < DIR_TOLERANCE &&
			std::abs(nDirY - dirY) < DIR_TOLERANCE;

		if( !gradientMatrix.at(ny, nx).visited &&
		    gradientMatrix.at(ny, nx).Threshold.lowerThreshold > 0.0 &&
		    similarDirection)
		{
			gradientMatrix.at(ny, nx).visited = true;
			gradientMatrix.at(ny, nx).binary = 255;
			Travers(gradientMatrix, ny, nx);
			return true;
		}
	}

	return false;
}

Matrix<uint8_t> Algorithms::calculateGradient(Matrix<uint8_t>& input) 
{
    Matrix<Gradient> gradientField(input.rows(), input.cols());
    input.iterate([&](uint8_t value, uint32_t y, uint32_t x) 
    {
        LOG_TRACE("Algorithms", "value " + TOS(value));
        double dx = .0;
        double dy = .0;
		/* calculate dx and dy */
        if( x - 1 >= 0 && x + 1 < input.cols() &&
            y - 1 >= 0 && y + 1 < input.rows() ) 
        {                  
	        gradientField.at(y, x).dx = input.at(y, x + 1) - value;
			gradientField.at(y, x).dy = input.at(y + 1, x) - value;
            // gradientField.at(y, x).dx = 0.5 * (input.at(y, x + 1) - input.at(y, x - 1));
            // gradientField.at(y, x).dy = 0.5 * (input.at(y + 1, x) - input.at(y - 1, x));
        }

        /* calculate the edge direction */          
        double angle = Algorithms::toDegrees( atan2(dy, dx) );
        static const int directions[4] = {0, 45, 90, 135};
        float absNum = std::fmod(std::abs(angle) + 180.0f, 180.0f); // normalize to [0, 180)

        for( int i = 0; i < 4; ++i ) 
        {
            float center = directions[i];
            if( absNum >= center - 22.5f && absNum < center + 22.5f ) 
            {
                gradientField.at(y, x).deg = directions[i];
                
		        switch( directions[i] ) 
		        {
                    case 0:
                        gradientField.at(y, x).dirX = 0;					
                        gradientField.at(y, x).dirY = 0;
                        break;
                    case 45:
                        gradientField.at(y, x).dirX = 1;
                        gradientField.at(y, x).dirY = -1;
                        break;
                    case 90:
                        gradientField.at(y, x).dirX  = 0;
                        gradientField.at(y, x).dirY = -1;
                        break;
                    case 135:
                        gradientField.at(y, x).dirX = -1;
                        gradientField.at(y, x).dirY = -1;
                        break;
                    default:
                        gradientField.at(y, x).dirX = 0;
                        gradientField.at(y, x).dirY = 0;
                        break;
                }
            }

            break;
        }

        /* Gradient length */
        gradientField.at(y, x).val = sqrt(gradientField.at(y, x).dx * gradientField.at(y, x).dx + gradientField.at(y, x).dy * gradientField.at(y, x).dy);
		
        uint8_t lower = 25;//3;
        uint8_t higher = 55;//40;
        
        /* Thresholding */
        if( abs(gradientField.at(y, x).val) < lower ) 
		{
            gradientField.at(y, x).val = 0.0;
            gradientField.at(y, x).Threshold.lowerThreshold = 0.0;
            gradientField.at(y, x).Threshold.higherThreshold = 0.0;
		} else if( abs(gradientField.at(y, x).val) >= lower && abs(gradientField.at(y, x).val) < higher )
		{
            gradientField.at(y, x).Threshold.lowerThreshold = abs(gradientField.at(y, x).val);
            gradientField.at(y, x).Threshold.higherThreshold = 0.0;
		} else if( abs(gradientField.at(y, x).val) >= higher )
		{
            gradientField.at(y, x).Threshold.lowerThreshold = 0.0;
            gradientField.at(y, x).Threshold.higherThreshold = abs(gradientField.at(y, x).val);
		}			
    });

    /* non maxima suppression */
    gradientField.iterate([&](Gradient& gradient, uint32_t y, uint32_t x) 
    {
		double anker, plusDir, minusDir;
		if( y-1 >= 0 && y+1 < gradientField.rows() ) 
		{
			if( x-1 >= 0 && x+1 < gradientField.cols() ) 
			{
				anker = gradient.val;
				plusDir = gradientField.at(y + gradient.dirX, x + gradient.dirY).val;
				minusDir = gradientField.at(y - gradient.dirY, x - gradient.dirY).val;

				if (plusDir > anker || minusDir > anker) 
					gradient.val = .0;
			}
		}
		else
			gradient.val = .0;
    });

    /* Hysteresis */ 
	gradientField.iterate([&](Gradient& gradient, uint32_t y, uint32_t x) 
    {
		// if( gradient.Threshold.higherThreshold > 0.0 )
			Travers(gradientField, y, x) ? gradientField.at(y, x).binary = 255 : gradientField.at(y, x).binary = 0;
	});

    Matrix<uint8_t> returnMat(input.rows(), input.cols());
    gradientField.iterate([&](Gradient& gradient, uint32_t y, uint32_t x) 
    {
        returnMat.at(y, x) = gradient.binary;
    });

    return returnMat;
}

FilterKernel Algorithms::gaussianKernel(const uint8_t& kernel_size)
{
    double sigma = std::max(1.0, kernel_size / 6.0);
    double s = 2.0 * sigma * sigma;
    int radius = (kernel_size - 1) / 2;

    FilterKernel kernel(kernel_size, std::vector<double>(kernel_size));
    double sum = 0.0;

    for (int x = -radius; x <= radius; ++x) 
    {
        for (int y = -radius; y <= radius; ++y) 
        {
            double r = std::sqrt(x * x + y * y);
            double value = std::exp(-(r * r) / s) / (M_PI * s);
            kernel[x + radius][y + radius] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernel_size; ++i) 
    {
        for (int j = 0; j < kernel_size; ++j) 
        {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

Matrix<uint8_t> Algorithms::gaussian(Matrix<uint8_t>& input, const uint8_t& kernel_size) 
{
    Matrix<float> floatInput(input.rows(), input.cols());
    input.iterate([&](uint8_t value, uint32_t y, uint32_t x) {
        floatInput.at(y, x) = static_cast<float>(value) * 0.001f;
    });
    
    Matrix<float> blurred = floatInput.convolve(gaussianKernel( kernel_size ));
    Matrix<uint8_t> result(blurred.rows(), blurred.cols());
    blurred.iterate([&](float value, uint32_t y, uint32_t x) {
        result.at(y, x) = static_cast<uint8_t>(value * 1000.0f);
    });

    return result;
}
