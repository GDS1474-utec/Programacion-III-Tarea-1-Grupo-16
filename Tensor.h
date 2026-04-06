#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <stdexcept>
using namespace std;

class TensorTransform;

class Tensor {
private:
    vector<size_t> shape;
    size_t total_size;
    double* data;

public:
    Tensor();
    Tensor(const vector<size_t>& shape, const vector<double>& values);
    Tensor(const vector<size_t>& shape, double* raw_data, size_t total_size);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);
    static Tensor concat(const vector<Tensor>& tensors, size_t dim);

    Tensor apply(const TensorTransform& transform) const;
    Tensor view(const vector<size_t>& new_shape);
    Tensor unsqueeze(size_t dim);

    const vector<size_t>& getShape() const;
    const double* getData() const;
    size_t getTotalSize() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    void printShape() const;
    void printData() const;
    void printRecursive(size_t dim, size_t& index) const;

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
};

#endif