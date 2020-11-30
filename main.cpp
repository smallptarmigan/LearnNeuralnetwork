#include <iostream>

#include "neuralnet.cpp"

void PrintVec(const std::vector<float> x){
    printf("[");
    for(int i = 0; i < x.size(); ++i){
        printf("%f", x[i]);
        if (i != x.size()-1){
            printf(" ");
        }
    }
    printf("]\n");
};

int main() {
    std::cout << "NN TEST PROGRAM" << std::endl;

    std::vector<float> X  = {1.0, 0.5};
    std::vector<float> W1 = {0.1, 0.3, 0.5, 0.2, 0.4, 0.6};
    std::vector<float> B1 = {0.1, 0.2, 0.3};
    std::vector<float> W2 = {0.1, 0.4, 0.2, 0.5, 0.3, 0.6};
    std::vector<float> B2 = {0.1, 0.2};
    std::vector<float> W3 = {0.1, 0.3, 0.2, 0.4};
    std::vector<float> B3 = {0.1, 0.2};

    // Init Parameters
    int EPOCH = 1;

    // Define layer
    nn::layer L1 = nn::layer(2, 3);
    nn::layer L2 = nn::layer(3, 2);
    nn::layer L3 = nn::layer(2, 2);

    // Activation function
    nn::activation ac;

    // Init W and B
    L1.setW(W1);
    L1.setB(B1);
    L2.setW(W2);
    L2.setB(B2);
    L3.setW(W3);
    L3.setB(B3);

    for (int i = 0; i < EPOCH; ++i){
        // Prediction
        L1.forword(X);
        ac.sigmoid(X);
        L2.forword(X);
        ac.sigmoid(X);
        L3.forword(X);
        ac.softmax(X);
        PrintVec(X);

    }

    return 0;
}

