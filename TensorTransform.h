#ifndef TENSORTRANSFORM_H
#define TENSORTRANSFORM_H

#include "Tensor.h"

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

#endif