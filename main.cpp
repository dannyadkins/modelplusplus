#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// this is the base class for all operations
class Op {
    public:
        virtual void forward() = 0;
        virtual void backward() = 0;
};

// activations 
class ReLU : public Op {
    public:
        vector<double> input;
        vector<double> output;
        vector<double> grad_input;
        vector<double> grad_output;

        ReLU(vector<double> input) {
            this->input = input;
        }

        void forward() {
            output.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = input[i] > 0 ? input[i] : 0;
            }
        }

        void backward() {
            grad_input.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                grad_input[i] = input[i] > 0 ? 1 : 0;
            }
        }
};

class Sigmoid : public Op {
    public:
        vector<double> input;
        vector<double> output;
        vector<double> grad_input;
        vector<double> grad_output;

        Sigmoid(vector<double> input) {
            this->input = input;
        }

        void forward() {
            output.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = 1.0 / (1.0 + exp(-input[i]));
            }
        }

        void backward() {
            grad_input.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                grad_input[i] = output[i] * (1.0 - output[i]);
            }
        }
};

// test relu and sigmoid
void test_activations() {
    vector<double> input = {1, 2, 3, 4, 5};
    ReLU relu(input);
    relu.forward();
    cout << "relu output: " << endl;
    for (auto& x : relu.output) {
        cout << x << " ";
    }
    cout << endl;
    relu.backward();
    for (auto& x : relu.grad_input) {
        cout << x << " ";
    }
    cout << endl;

    Sigmoid sigmoid(input);
    sigmoid.forward();
    cout << "sigmoid output: " << endl;
    for (auto& x : sigmoid.output) {
        cout << x << " ";
    }
    cout << endl;
    sigmoid.backward();
    for (auto& x : sigmoid.grad_input) {
        cout << x << " ";
    }
    cout << endl;
}

// main func
int main() {
    test_activations();
    return 0;
}