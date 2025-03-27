#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

class TinyNN
{
private:
    // Weights [2][64]
    const float weights[2][64] = {
        {0.0276, 0.0342, 0.0698, 0.0553, 0.0346, -0.1244, 0.0333, -0.0364,
         0.1203, 0.1366, -0.0452, -0.0659, -0.0602, -0.0102, -0.1255, 0.0720,
         -0.2111, -0.0472, -0.0921, 0.0524, -0.0042, 0.1133, 0.1604, -0.0044,
         0.0570, 0.0359, 0.0800, 0.1479, 0.0092, 0.0010, 0.2558, -0.0194,
         -0.1015, -0.0216, -0.1029, -0.2129, -0.0465, 0.0378, -0.0337, -0.1557,
         0.0395, -0.0367, 0.0684, 0.0012, 0.0432, -0.0336, 0.0577, 0.0155,
         -0.0515, 0.0730, -0.0713, 0.0627, -0.0290, -0.0547, 0.1320, -0.1603,
         -0.0533, -0.0285, 0.0431, -0.0126, 0.1643, 0.1481, 0.0342, 0.0600},
        {-0.0429, -0.0885, -0.0750, -0.1268, 0.0148, -0.0026, -0.0407, -0.0359,
         0.0905, -0.1191, 0.1392, 0.0802, 0.0140, -0.0542, 0.0083, -0.0335,
         0.0185, 0.0248, 0.0039, -0.1762, -0.0201, 0.0749, -0.1363, 0.0764,
         0.0705, -0.0197, -0.0644, -0.1696, 0.0394, -0.0382, -0.1668, 0.0465,
         0.0802, -0.0765, -0.0004, 0.0674, -0.0486, -0.0940, -0.0182, 0.0943,
         0.0939, -0.0948, -0.0226, -0.1499, 0.0635, -0.0339, 0.1707, 0.1634,
         0.1219, -0.0531, 0.0605, 0.0455, 0.0368, -0.0956, -0.0049, -0.0773,
         -0.1261, -0.1225, 0.0894, -0.1316, 0.0020, -0.1088, -0.0693, -0.0681}};

    // Biases [2]
    const float biases[2] = {0.0519, 0.0504};

public:
    // Preprocess image (convert to grayscale, normalize)
    std::vector<float> preprocess_image(const unsigned char *img_data, int width, int height, int channels)
    {
        std::vector<float> processed(64, 0.0f);

        // Simple average pooling to resize to 8x8 if needed
        int block_w = width / 8;
        int block_h = height / 8;

        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                float sum = 0.0f;
                int count = 0;

                // Average pixels in the block
                for (int by = 0; by < block_h; by++)
                {
                    for (int bx = 0; bx < block_w; bx++)
                    {
                        int px = x * block_w + bx;
                        int py = y * block_h + by;

                        if (px < width && py < height)
                        {
                            float pixel = 0.0f;
                            if (channels == 1)
                            {
                                pixel = img_data[py * width + px] / 255.0f;
                            }
                            else
                            {
                                // Convert RGB to grayscale
                                int idx = (py * width + px) * channels;
                                float r = img_data[idx] / 255.0f;
                                float g = img_data[idx + 1] / 255.0f;
                                float b = img_data[idx + 2] / 255.0f;
                                pixel = 0.299f * r + 0.587f * g + 0.114f * b;
                            }
                            sum += pixel;
                            count++;
                        }
                    }
                }

                if (count > 0)
                {
                    // Normalize to [-0.5, 0.5] range (same as Python)
                    processed[y * 8 + x] = (sum / count) - 0.5f;
                }
            }
        }

        return processed;
    }

    // Forward pass function
    std::vector<float> predict(const std::vector<float> &input)
    {
        if (input.size() != 64)
        {
            throw std::invalid_argument("Input must be 64 elements (8x8 image flattened)");
        }

        std::vector<float> output(2, 0.0f);

        for (int i = 0; i < 2; i++)
        {
            output[i] = biases[i];
            for (int j = 0; j < 64; j++)
            {
                output[i] += input[j] * weights[i][j];
            }
        }

        return output;
    }

    // Helper function to get class prediction (0 or 1)
    int predict_class(const std::vector<float> &input)
    {
        auto output = predict(input);
        return (output[0] > output[1]) ? 0 : 1;
    }

    // Load image and make prediction
    int predict_image(const char *filename)
    {
        int width, height, channels;
        unsigned char *img_data = stbi_load(filename, &width, &height, &channels, 0);

        if (!img_data)
        {
            throw std::runtime_error("Failed to load image");
        }

        try
        {
            auto processed = preprocess_image(img_data, width, height, channels);
            stbi_image_free(img_data);
            return predict_class(processed);
        }
        catch (...)
        {
            stbi_image_free(img_data);
            throw;
        }
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image.jpg>" << std::endl;
        return 1;
    }

    TinyNN network;

    try
    {
        int prediction = network.predict_image(argv[1]);
        std::cout << "Predicted class: " << prediction
                  << " (0 for digit 1, 1 for digit 2)" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}