//
// Created by issei on 2020/10/09.
//
#include <iostream>
#include <cassert>
#include <vector>

namespace nn {
    class layer {
    private:
        int input_d;
        int output_d;
        std::vector<float> W;
        std::vector<float> B;

    public:
        // Constructor
        layer(const int in_features, const int out_features){
            input_d = in_features;
            output_d = out_features;
            W.resize(in_features * out_features);
            B.resize(out_features);
        };

        // Deconstructor
        ~layer(){
            std::vector<float>().swap(W);
            std::vector<float>().swap(B);
        };

        // Forword
        void forword(std::vector<float>& y) {
            std::vector<float> x(y.size());
            for(int i = 0; i < y.size(); ++i){ x[i] = y[i]; }
            y.resize(output_d);
            for(int i = 0; i < output_d; ++i){
                float row = 0;
                for(int j = 0; j < input_d; ++j){
                    row += x[j] * W[j * output_d + i];
                }
                y[i] = row + B[i];
            }
        }

        void setW(const std::vector<float> w){
            assert (w.size() == W.size());
            for(int i = 0; i < W.size(); ++i){
                W[i] = w[i];
            }
        }

        void setB(const std::vector<float> b){
            assert (b.size() == B.size());
            for(int i = 0; i < B.size(); ++i){
                B[i] = b[i];
            }
        }

        // Get methods
        std::vector<float> getW(){return W;}
        std::vector<float> getB(){return B;}
    };

    class activation {
    private:
        float max(const std::vector<float>& x){
            float a = -FLT_MAX;
            for(int i = 0; i < x.size(); ++i){
                if(x[i] > a) { a = x[i]; }
            }
            return a;
        }

    public:
        void sigmoid(std::vector<float>& x){
            for(int i = 0; i < x.size(); ++i){
                x[i] = 1.0 / (1.0 + std::exp(-1.0 * x[i]));
            }
        }

        void softmax(std::vector<float>& x){
            float xsum = 0;
            for(int i = 0; i < x.size(); ++i){
                x[i] = std::exp(x[i] - max(x));
                xsum += x[i];
            }
            for(int i = 0; i < x.size(); ++i){
                x[i] = x[i] / xsum;
            }
        }
    };
}

