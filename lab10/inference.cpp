#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>


using half = __fp16;


inline half half_relu(half x)
{
    float fx = static_cast<float>(x);
    float out_f = (fx > 0.0f) ? fx : 0.0f;
    return static_cast<half>(out_f);
}

int main()
{

    // 8*8 0/1 image will be input so 64 inputs
    // Number of weights =   number_of_hidden_neurons*number_of_input  + number_of_output_neurons*number_of_hidden_neurons
    // Number of hidden neurons = 3*64 + 10*3 = 192


    // FOR NOW I have set the code to have random  values for weights and inputs, but you can have a them 
    // sent to netfpga via payload or hard coded to your ROM

    // SECTION -1  
    //Ignore this for now, just random values initialized for weighsts and inputs
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

   
    const int INPUT_SIZE  = 64;
    const int HIDDEN_SIZE = 3;
    const int OUTPUT_SIZE = 10;

   
    half input[INPUT_SIZE];
    for(int i = 0; i < INPUT_SIZE; i++) {
        // random [0,1)
        float val = static_cast<float>(std::rand()) / RAND_MAX;
        input[i] = static_cast<half>(val);
    }


    half hidden_weights[HIDDEN_SIZE][INPUT_SIZE];
    for(int h = 0; h < HIDDEN_SIZE; h++) {
        for(int j = 0; j < INPUT_SIZE; j++) {
            // random in [-1, 1]
            float val = 2.0f * (static_cast<float>(std::rand()) / RAND_MAX) - 1.0f;
            hidden_weights[h][j] = static_cast<half>(val);
        }
    }

    half output_weights[OUTPUT_SIZE][HIDDEN_SIZE];
    for(int k = 0; k < OUTPUT_SIZE; k++) {
        for(int h = 0; h < HIDDEN_SIZE; h++) {
            float val = 2.0f * (static_cast<float>(std::rand()) / RAND_MAX) - 1.0f;
            output_weights[k][h] = static_cast<half>(val);
        }
    }
 
    //  Ignore code section done



    //  Number of hidden layers = 3
    //  You have 4 threads so unroll the outer loop 3 times to run each neuron per thread

    half hidden[HIDDEN_SIZE];


    //SECTION 2: 
    for(int h = 0; h < HIDDEN_SIZE; h++) {
        float sum = 0.0f;
        for(int j = 0; j < INPUT_SIZE; j++) {
            sum += static_cast<float>(hidden_weights[h][j]) *
                   static_cast<float>(input[j]);
        }
        hidden[h] = half_relu(static_cast<half>(sum));
    }

    // Have a shared semmaphore across three threads to synchronize the threads(each thread increments the semaphore)
    //Unrolling end

    // When all the three threds complete (when you read the semaphore vaue as 3) then one of the thread T0 (lets call it a host code thread) will execute the next steps
    half outputs[OUTPUT_SIZE];
    //SECTION 3
    for(int k = 0; k < OUTPUT_SIZE; k++) {
        float sum = 0.0f;
        for(int h_ = 0; h_ < HIDDEN_SIZE; h_++) {
            sum += static_cast<float>(output_weights[k][h_]) *
                   static_cast<float>(hidden[h_]);
        }
        outputs[k] = static_cast<half>(sum); // raw logits (no activation)
    }


    std::cout << "Output logits (printed as floats, stored in half):\n";
    for(int k = 0; k < OUTPUT_SIZE; k++) {
        // cast to float just for printing
        float val = static_cast<float>(outputs[k]);
        std::cout << val << "  ";
    }
    std::cout << "\n";

    int best_class = 0;
    float best_val = static_cast<float>(outputs[0]);
    for(int k = 1; k < OUTPUT_SIZE; k++) {
        float val = static_cast<float>(outputs[k]);
        if(val > best_val) {
            best_val = val;
            best_class = k;
        }
    }
    std::cout << "Predicted class: " << best_class << "\n";

    return 0;
}
