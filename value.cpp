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

    std::cout << "grad test passed" << std::endl;
}

// main func
int main() {
    test_grad();
    return 0;
}