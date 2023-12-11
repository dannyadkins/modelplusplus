#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <set>
#include <cassert>

class Value {
public:
    float data;
    float grad;
    std::function<void()> _backward;
    std::set<std::shared_ptr<Value>> _prev;
    std::string _op;

    Value(float data) : data(data), grad(0), _backward([](){}), _op("") {}

    Value(float data, std::vector<std::shared_ptr<Value>> children, std::string op) 
    : data(data), grad(0), _backward([](){}), _op(op) {
        for (auto& child : children) {
            _prev.insert(child);
        }
    }

    std::shared_ptr<Value> create_shared() {
        return std::make_shared<Value>(*this);
    }

    static std::shared_ptr<Value> add(std::shared_ptr<Value> self, std::shared_ptr<Value> other) {
        auto out = std::make_shared<Value>(self->data + other->data, std::vector<std::shared_ptr<Value>>{self, other}, "+");

        out->_backward = [self, other, out]() {
            self->grad += out->grad;
            other->grad += out->grad;
        };

        return out;
    }

    static std::shared_ptr<Value> multiply(std::shared_ptr<Value> self, std::shared_ptr<Value> other) {
        auto out = std::make_shared<Value>(self->data * other->data, std::vector<std::shared_ptr<Value>>{self, other}, "*");

        out->_backward = [self, other, out]() {
            self->grad += other->data * out->grad;
            other->grad += self->data * out->grad;
        };

        return out;
    }

    void backward(std::shared_ptr<Value> self) {
        // topsort order 
        std::vector<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>> visited;
        std::function<void(std::shared_ptr<Value>)> build_topo;
        build_topo = [&topo, &visited, &build_topo](std::shared_ptr<Value> v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (auto& child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };
        
        build_topo(self);

        grad = 1;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

};

class Module {
public:
    virtual void zero_grad() {
        for (auto& p : parameters()) {
            p->grad = 0;
        }
    }

    virtual std::vector<std::shared_ptr<Value>> parameters() {
        return {};
    }
};

class Neuron : public Module {
public:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

    Neuron(int nin, bool nonlin=true) : nonlin(nonlin) {
        for (int i = 0; i < nin; ++i) {
            w.push_back(std::make_shared<Value>(1.0));
        }
        b = std::make_shared<Value>(0.0);
    }

    std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> x) {
        auto act = b;
        for (int i = 0; i < x.size(); ++i) {
            act = Value::add(act, Value::multiply(w[i], x[i]));
        }
        return nonlin ? act : act;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> out;
        for (auto& wi : w) {
            out.push_back(wi);
        }
        out.push_back(b);
        return out;
    }

    std::string repr() {
        return "Neuron";
    }
};

class Layer : public Module {
public:
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer(int nin, int nout, bool nonlin=true) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(std::make_shared<Neuron>(nin, nonlin));
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x) {
        std::vector<std::shared_ptr<Value>> out;
        for (auto& neuron : neurons) {
            out.push_back((*neuron)(x));
        }
        return out;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> out;
        for (auto& neuron : neurons) {
            for (auto& p : neuron->parameters()) {
                out.push_back(p);
            }
        }
        return out;
    }

    std::string repr() {
        return "Layer";
    }
};

class MLP : public Module {
public:
    std::vector<std::shared_ptr<Layer>> layers;

    MLP(int nin, std::vector<int> nouts) {
        std::vector<int> sz;
        sz.push_back(nin);
        for (auto& nout : nouts) {
            sz.push_back(nout);
        }
        for (int i = 0; i < nouts.size(); ++i) {
            layers.push_back(std::make_shared<Layer>(sz[i], sz[i+1], i != nouts.size()-1));
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x) {
        for (auto& layer : layers) {
            x = (*layer)(x);
        }
        return x;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> out;
        for (auto& layer : layers) {
            for (auto& p : layer->parameters()) {
                out.push_back(p);
            }
        }
        return out;
    }

    std::string repr() {
        return "MLP";
    }
};

// test grad calculation
void test_grad() {
    auto a = std::make_shared<Value>(1.0);
    auto b = std::make_shared<Value>(2.0);
    auto c = std::make_shared<Value>(3.0);
    auto d = std::make_shared<Value>(4.0);

    auto e = Value::add(a, b);
    auto f = Value::multiply(c, d);
    auto g = Value::add(e, f);

    g->backward(g);

    assert(a->grad == 1);
    assert(b->grad == 1);
    assert(c->grad == 4);
    assert(d->grad == 3);

    std::cout << "Passed: test_grad" << std::endl;
}

void test_num_params() {
    auto mlp = MLP(2, {16, 16, 1});
    std::cout << mlp.repr() << std::endl;
    assert(mlp.parameters().size() == 2*16 + 16*16 + 16*1 + 16 + 16 + 1);
    std::cout << "Passed: test_num_params" << std::endl;
}

void test_mlp() {
    auto mlp = MLP(2, {3, 1});
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1.0), std::make_shared<Value>(2.0)};
    auto y = mlp(x);
    y[0]->backward(y[0]); 
}

// WIP 
std::shared_ptr<Value> loss(std::vector<std::shared_ptr<Value>> X, std::vector<std::shared_ptr<Value>> y, std::shared_ptr<MLP> model, int batch_size) {

    std::vector<std::vector<std::shared_ptr<Value>>> inputs;
    for (auto& xrow : X) {
        std::vector<std::shared_ptr<Value>> row;
        row.push_back(xrow);
        inputs.push_back(row);
    }
    // print 
    std::cout << "Inputs: " << std::endl;
    for (auto& input : inputs) {
        std::cout << input[0]->data << std::endl;
    }

    // forward the model to get scores
    std::vector<std::shared_ptr<Value>> scores;
    for (auto& input : inputs) {
        scores.push_back(model->operator()(input)[0]);
    }
    // print scores
    std::cout << "Scores: " << std::endl;
    for (auto& score : scores) {
        std::cout << score->data << std::endl;
    }

    // svm "max-margin" loss
    std::vector<std::shared_ptr<Value>> losses;
    for (int i = 0; i < y.size(); ++i) {
        losses.push_back(Value::add(std::make_shared<Value>(1.0), Value::multiply(y[i], scores[i])));
    }
    std::shared_ptr<Value> data_loss = std::make_shared<Value>(0.0);
    for (auto& lossi : losses) {
        data_loss = Value::add(data_loss, lossi);
    }
    data_loss = Value::multiply(data_loss, std::make_shared<Value>(1.0 / losses.size()));
    // L2 regularization
    float alpha = 1e-4;
    std::shared_ptr<Value> reg_loss = std::make_shared<Value>(0.0);
    for (auto& p : model->parameters()) {
        reg_loss = Value::add(reg_loss, Value::multiply(p, p));
    }
    reg_loss = Value::multiply(reg_loss, std::make_shared<Value>(alpha));
    std::shared_ptr<Value> total_loss = Value::add(data_loss, reg_loss);

    // also get accuracy
    std::vector<std::shared_ptr<Value>> accuracy;
    for (int i = 0; i < y.size(); ++i) {
        accuracy.push_back(Value::add(std::make_shared<Value>(y[i]->data > 0), std::make_shared<Value>(scores[i]->data > 0)));
    }
    std::shared_ptr<Value> acc = std::make_shared<Value>(0.0);
    for (auto& acci : accuracy) {
        acc = Value::add(acc, acci);
    }
    acc = Value::multiply(acc, std::make_shared<Value>(1.0 / accuracy.size()));
    
    return total_loss;
}

void test_loss() {
    auto mlp = MLP(2, {16, 16, 1});
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1.0), std::make_shared<Value>(2.0)};
    auto y = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1.0)};
    auto l = loss(x, y, std::make_shared<MLP>(mlp), -1);
    l->backward(l);
    // print loss
    std::cout << l->data << std::endl;
    std::cout << "Passed: test_loss" << std::endl;
}


// main func
int main() {
    test_grad();
    test_num_params();
    test_mlp();
    test_loss();
    return 0;
}