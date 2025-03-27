#include <math.h>

// Neural network weights and biases (32-bit floats)
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

const float biases[2] = {0.0519, 0.0504};

// Neural network computation function
// Assumes input is a 64-element array of 32-bit floats in memory
// Returns 0 for digit 1, 1 for digit 2
int tiny_nn_predict(const float *input)
{
    float output[2] = {biases[0], biases[1]};

    // Output 0 calculations (64 multiply-accumulate operations)
    output[0] += input[0] * weights[0][0];
    output[0] += input[1] * weights[0][1];
    output[0] += input[2] * weights[0][2];
    output[0] += input[3] * weights[0][3];
    output[0] += input[4] * weights[0][4];
    output[0] += input[5] * weights[0][5];
    output[0] += input[6] * weights[0][6];
    output[0] += input[7] * weights[0][7];
    output[0] += input[8] * weights[0][8];
    output[0] += input[9] * weights[0][9];
    output[0] += input[10] * weights[0][10];
    output[0] += input[11] * weights[0][11];
    output[0] += input[12] * weights[0][12];
    output[0] += input[13] * weights[0][13];
    output[0] += input[14] * weights[0][14];
    output[0] += input[15] * weights[0][15];
    output[0] += input[16] * weights[0][16];
    output[0] += input[17] * weights[0][17];
    output[0] += input[18] * weights[0][18];
    output[0] += input[19] * weights[0][19];
    output[0] += input[20] * weights[0][20];
    output[0] += input[21] * weights[0][21];
    output[0] += input[22] * weights[0][22];
    output[0] += input[23] * weights[0][23];
    output[0] += input[24] * weights[0][24];
    output[0] += input[25] * weights[0][25];
    output[0] += input[26] * weights[0][26];
    output[0] += input[27] * weights[0][27];
    output[0] += input[28] * weights[0][28];
    output[0] += input[29] * weights[0][29];
    output[0] += input[30] * weights[0][30];
    output[0] += input[31] * weights[0][31];
    output[0] += input[32] * weights[0][32];
    output[0] += input[33] * weights[0][33];
    output[0] += input[34] * weights[0][34];
    output[0] += input[35] * weights[0][35];
    output[0] += input[36] * weights[0][36];
    output[0] += input[37] * weights[0][37];
    output[0] += input[38] * weights[0][38];
    output[0] += input[39] * weights[0][39];
    output[0] += input[40] * weights[0][40];
    output[0] += input[41] * weights[0][41];
    output[0] += input[42] * weights[0][42];
    output[0] += input[43] * weights[0][43];
    output[0] += input[44] * weights[0][44];
    output[0] += input[45] * weights[0][45];
    output[0] += input[46] * weights[0][46];
    output[0] += input[47] * weights[0][47];
    output[0] += input[48] * weights[0][48];
    output[0] += input[49] * weights[0][49];
    output[0] += input[50] * weights[0][50];
    output[0] += input[51] * weights[0][51];
    output[0] += input[52] * weights[0][52];
    output[0] += input[53] * weights[0][53];
    output[0] += input[54] * weights[0][54];
    output[0] += input[55] * weights[0][55];
    output[0] += input[56] * weights[0][56];
    output[0] += input[57] * weights[0][57];
    output[0] += input[58] * weights[0][58];
    output[0] += input[59] * weights[0][59];
    output[0] += input[60] * weights[0][60];
    output[0] += input[61] * weights[0][61];
    output[0] += input[62] * weights[0][62];
    output[0] += input[63] * weights[0][63];

    // Output 1 calculations (64 multiply-accumulate operations)
    output[1] += input[0] * weights[1][0];
    output[1] += input[1] * weights[1][1];
    output[1] += input[2] * weights[1][2];
    output[1] += input[3] * weights[1][3];
    output[1] += input[4] * weights[1][4];
    output[1] += input[5] * weights[1][5];
    output[1] += input[6] * weights[1][6];
    output[1] += input[7] * weights[1][7];
    output[1] += input[8] * weights[1][8];
    output[1] += input[9] * weights[1][9];
    output[1] += input[10] * weights[1][10];
    output[1] += input[11] * weights[1][11];
    output[1] += input[12] * weights[1][12];
    output[1] += input[13] * weights[1][13];
    output[1] += input[14] * weights[1][14];
    output[1] += input[15] * weights[1][15];
    output[1] += input[16] * weights[1][16];
    output[1] += input[17] * weights[1][17];
    output[1] += input[18] * weights[1][18];
    output[1] += input[19] * weights[1][19];
    output[1] += input[20] * weights[1][20];
    output[1] += input[21] * weights[1][21];
    output[1] += input[22] * weights[1][22];
    output[1] += input[23] * weights[1][23];
    output[1] += input[24] * weights[1][24];
    output[1] += input[25] * weights[1][25];
    output[1] += input[26] * weights[1][26];
    output[1] += input[27] * weights[1][27];
    output[1] += input[28] * weights[1][28];
    output[1] += input[29] * weights[1][29];
    output[1] += input[30] * weights[1][30];
    output[1] += input[31] * weights[1][31];
    output[1] += input[32] * weights[1][32];
    output[1] += input[33] * weights[1][33];
    output[1] += input[34] * weights[1][34];
    output[1] += input[35] * weights[1][35];
    output[1] += input[36] * weights[1][36];
    output[1] += input[37] * weights[1][37];
    output[1] += input[38] * weights[1][38];
    output[1] += input[39] * weights[1][39];
    output[1] += input[40] * weights[1][40];
    output[1] += input[41] * weights[1][41];
    output[1] += input[42] * weights[1][42];
    output[1] += input[43] * weights[1][43];
    output[1] += input[44] * weights[1][44];
    output[1] += input[45] * weights[1][45];
    output[1] += input[46] * weights[1][46];
    output[1] += input[47] * weights[1][47];
    output[1] += input[48] * weights[1][48];
    output[1] += input[49] * weights[1][49];
    output[1] += input[50] * weights[1][50];
    output[1] += input[51] * weights[1][51];
    output[1] += input[52] * weights[1][52];
    output[1] += input[53] * weights[1][53];
    output[1] += input[54] * weights[1][54];
    output[1] += input[55] * weights[1][55];
    output[1] += input[56] * weights[1][56];
    output[1] += input[57] * weights[1][57];
    output[1] += input[58] * weights[1][58];
    output[1] += input[59] * weights[1][59];
    output[1] += input[60] * weights[1][60];
    output[1] += input[61] * weights[1][61];
    output[1] += input[62] * weights[1][62];
    output[1] += input[63] * weights[1][63];

    return (output[0] > output[1]) ? 0 : 1;
}

// Example memory interface for FPGA
// This would be replaced with your actual FPGA memory access
int main()
{
    // Example input buffer - in real FPGA this would be memory-mapped
    float input_buffer[64];

    // Initialize with test data (would be filled from memory in real deployment)
    for (int i = 0; i < 64; i++)
    {
        input_buffer[i] = 0.0f; // Replace with actual image data
    }

    // Compute prediction
    int prediction = tiny_nn_predict(input_buffer);

    return prediction;
}