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

void test_mlp() {
    auto mlp = MLP(2, {3, 1});
    auto x = std::vector<std::shared_ptr<Value>>{std::make_shared<Value>(1.0), std::make_shared<Value>(2.0)};
    auto y = mlp(x);
    y[0]->backward(y[0]); 
}


// main func
int main() {
    test_grad();
    test_mlp();
    return 0;
}