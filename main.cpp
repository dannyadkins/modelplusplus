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

// tensor with graph data needed for backprop
class Tensor {
    public:
        vector<double> data;
        vector<double> grad;
        vector<Tensor*> children;
        vector<Tensor*> parents;
        Op* op;

        Tensor(vector<double> data) {
            this->data = data;
        }

        void backward() {
            grad.resize(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                grad[i] = 1.0;
            }
            for (auto& child : children) {
                for (size_t i = 0; i < data.size(); ++i) {
                    grad[i] *= child->grad[i];
                }
            }
            op->backward();
            for (auto& parent : parents) {
                for (size_t i = 0; i < data.size(); ++i) {
                    parent->grad[i] += grad[i];
                }
            }
        }
};

// test tensor with assertions of correctness
void test_tensor() {
    vector<double> data = {1, 2, 3, 4, 5};
    Tensor tensor(data);
    ReLU relu(data);
    tensor.op = &relu;
    tensor.backward();
    cout << "tensor grad: " << endl;
    for (auto& x : tensor.grad) {
        cout << x << " ";
    }
    cout << endl;
}

// layers
class DenseLayer {
    public:
        vector<vector<double>> weights;
        vector<double> biases;
        vector<double> output;
        vector<double> gradients;
        vector<Tensor*> inputs;
        vector<Tensor*> outputs;
        Op* op;

        DenseLayer(vector<vector<double>> weights, vector<double> biases) {
            this->weights = weights;
            this->biases = biases;
        }

        void forward() {
            output.resize(weights.size());
            for (size_t i = 0; i < weights.size(); ++i) {
                output[i] = 0.0;
                for (size_t j = 0; j < inputs.size(); ++j) {
                    output[i] += weights[i][j] * inputs[j]->data[i];
                }
                output[i] += biases[i];
            }
            op->forward();
        }

        void backward() {
            gradients.resize(weights[0].size(), 0.0);
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[0].size(); ++j) {
                    gradients[j] += outputs[i]->grad[j] * weights[i][j];
                    weights[i][j] -= outputs[i]->grad[j] * inputs[j]->data[i];
                }
                biases[i] -= outputs[i]->grad[i];
            }
            op->backward();
            for (size_t i = 0; i < inputs.size(); ++i) {
                for (size_t j = 0; j < inputs[0]->data.size(); ++j) {
                    inputs[i]->grad[j] += gradients[j];
                }
            }
        }
};

// test dense layer
void test_dense_layer() {
    vector<vector<double>> weights = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
    vector<double> biases = {1, 2};
    vector<double> input = {1, 2, 3, 4, 5};
    vector<double> output = {1, 2};
    vector<double> grad_output = {1, 1};
    Tensor tensor_input(input);
    Tensor tensor_output(output);
    Tensor tensor_grad_output(grad_output);
    vector<Tensor*> inputs = {&tensor_input};
    vector<Tensor*> outputs = {&tensor_output};
    vector<Tensor*> grad_outputs = {&tensor_grad_output};
    DenseLayer dense_layer(weights, biases);
    ReLU relu(output);
    dense_layer.inputs = inputs;
    dense_layer.outputs = outputs;
    dense_layer.op = &relu;

    dense_layer.forward();
    dense_layer.backward();
    cout << "dense layer weights: " << endl;
    for (auto& row : dense_layer.weights) {
        for (auto& w : row) {
            cout << w << " ";
        }
        cout << endl;
    }
    cout << "dense layer biases: " << endl;
    for (auto& b : dense_layer.biases) {
        cout << b << " ";
    }
    cout << endl;
    cout << "dense layer gradients: " << endl;
    for (auto& g : dense_layer.gradients) {
        cout << g << " ";
    }
    cout << endl;
    cout << "tensor input grad: " << endl;
    for (auto& g : tensor_input.grad) {
        cout << g << " ";
    }
    cout << endl;
}

// main func
int main() {
    test_activations();
    test_tensor();
    test_dense_layer();
    return 0;
}