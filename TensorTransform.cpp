#include "TensorTransform.h"
#include <vector>
#include <cmath>
using namespace std;

Tensor ReLU::apply(const Tensor& t) const {
    vector<double> values;

    for (size_t i = 0; i < t.getTotalSize(); i++) {
        double x = t.getData()[i];

        if (x < 0) {
            values.push_back(0);
        } else {
            values.push_back(x);
        }
    }

    return Tensor(t.getShape(), values);
}

Tensor Sigmoid::apply(const Tensor& t) const {
    vector<double> values;

    for (size_t i = 0; i < t.getTotalSize(); i++) {
        double x = t.getData()[i];
        values.push_back(1.0 / (1.0 + exp(-x)));
    }

    return Tensor(t.getShape(), values);
}