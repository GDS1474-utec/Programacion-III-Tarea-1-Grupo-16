#include "Tensor.h"
#include "TensorTransform.h"
#include <random>
#include <iomanip>

// calcula tamaño total del tensor
static size_t calculate_total_size(const vector<size_t>& shape) {
    if (shape.empty() || shape.size() > 3) {
        throw invalid_argument("Shape invalido");
    }

    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw invalid_argument("No se permiten dimensiones 0");
        }
        total *= dim;
    }
    return total;
}

// constructor vacío
Tensor::Tensor() : total_size(0), data(nullptr) {}

// constructor con valores
Tensor::Tensor(const vector<size_t>& shape, const vector<double>& values) {
    this->shape = shape;
    total_size = calculate_total_size(shape);

    if (values.size() != total_size) {
        throw invalid_argument("Cantidad de valores incorrecta");
    }

    data = new double[total_size];
    for (size_t i = 0; i < total_size; i++) {
        data[i] = values[i];
    }
}

// constructor con puntero directo (sin copia)
Tensor::Tensor(const vector<size_t>& shape, double* raw_data, size_t total_size) {
    this->shape = shape;
    this->total_size = total_size;
    this->data = raw_data;
}

// constructor copia (deep copy)
Tensor::Tensor(const Tensor& other) {
    shape = other.shape;
    total_size = other.total_size;
    data = (total_size == 0 ? nullptr : new double[total_size]);

    for (size_t i = 0; i < total_size; i++) {
        data[i] = other.data[i];
    }
}

// operador asignación copia
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        delete[] data;

        shape = other.shape;
        total_size = other.total_size;
        data = (total_size == 0 ? nullptr : new double[total_size]);

        for (size_t i = 0; i < total_size; i++) {
            data[i] = other.data[i];
        }
    }
    return *this;
}

// constructor movimiento
Tensor::Tensor(Tensor&& other) noexcept {
    shape = other.shape;
    total_size = other.total_size;
    data = other.data;

    other.data = nullptr;
    other.total_size = 0;
    other.shape.clear();
}

// operador asignación movimiento
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        delete[] data;

        shape = other.shape;
        total_size = other.total_size;
        data = other.data;

        other.data = nullptr;
        other.total_size = 0;
        other.shape.clear();
    }
    return *this;
}

// destructor
Tensor::~Tensor() {
    delete[] data;
}

// crea tensor de ceros
Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total = calculate_total_size(shape);
    vector<double> values(total, 0.0);
    return Tensor(shape, values);
}

// crea tensor de unos
Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t total = calculate_total_size(shape);
    vector<double> values(total, 1.0);
    return Tensor(shape, values);
}

// crea tensor aleatorio
Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    if (min >= max) {
        throw invalid_argument("Rango invalido para random");
    }

    size_t total = calculate_total_size(shape);
    vector<double> values(total);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(min, max);

    for (size_t i = 0; i < total; i++) {
        values[i] = dist(gen);
    }

    return Tensor(shape, values);
}

// crea tensor secuencial
Tensor Tensor::arange(int start, int end) {
    if (start >= end) {
        throw invalid_argument("Rango invalido en arange");
    }

    vector<double> values;
    for (int i = start; i < end; i++) {
        values.push_back(i);
    }

    vector<size_t> shape = {values.size()};
    return Tensor(shape, values);
}

// concatena tensores
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
        throw invalid_argument("No hay tensores para concatenar");
    }

    vector<size_t> base_shape = tensors[0].shape;
    if (dim >= base_shape.size()) {
        throw invalid_argument("Dimension invalida en concat");
    }

    size_t new_dim_size = 0;
    for (const Tensor& t : tensors) {
        if (t.shape.size() != base_shape.size()) {
            throw invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones");
        }

        for (size_t i = 0; i < base_shape.size(); i++) {
            if (i != dim && t.shape[i] != base_shape[i]) {
                throw invalid_argument("Shapes incompatibles para concat");
            }
        }
        new_dim_size += t.shape[dim];
    }

    vector<size_t> new_shape = base_shape;
    new_shape[dim] = new_dim_size;
    size_t total = calculate_total_size(new_shape);
    double* new_data = new double[total];

    size_t index = 0;
    for (const Tensor& t : tensors) {
        for (size_t i = 0; i < t.total_size; i++) {
            new_data[index++] = t.data[i];
        }
    }

    return Tensor(new_shape, new_data, total);
}

// aplica transformación (polimorfismo)
Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

// cambia forma sin copiar (move)
Tensor Tensor::view(const vector<size_t>& new_shape) {
    size_t new_total = calculate_total_size(new_shape);
    if (new_total != total_size) {
        throw invalid_argument("view incompatible: cambia la cantidad de elementos");
    }

    Tensor result(new_shape, data, total_size);
    data = nullptr;
    total_size = 0;
    shape.clear();
    return result;
}

// agrega dimensión de tamaño 1
Tensor Tensor::unsqueeze(size_t dim) {
    if (shape.size() >= 3) {
        throw invalid_argument("unsqueeze excede el maximo de 3 dimensiones");
    }
    if (dim > shape.size()) {
        throw invalid_argument("Posicion invalida para unsqueeze");
    }

    vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);

    Tensor result(new_shape, data, total_size);
    data = nullptr;
    total_size = 0;
    shape.clear();
    return result;
}

// getters
const vector<size_t>& Tensor::getShape() const {
    return shape;
}

const double* Tensor::getData() const {
    return data;
}

size_t Tensor::getTotalSize() const {
    return total_size;
}

// suma de tensores
Tensor Tensor::operator+(const Tensor& other) const {
    vector<double> values;

    if (shape == other.shape) {
        for (size_t i = 0; i < total_size; i++) {
            values.push_back(data[i] + other.data[i]);
        }
        return Tensor(shape, values);
    }

    if (shape.size() == 2 && other.shape.size() == 2 && other.shape[0] == 1 && shape[1] == other.shape[1]) {
        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                values.push_back(data[i * shape[1] + j] + other.data[j]);
            }
        }
        return Tensor(shape, values);
    }

    if (shape.size() == 2 && other.shape.size() == 2 && shape[0] == 1 && shape[1] == other.shape[1]) {
        for (size_t i = 0; i < other.shape[0]; i++) {
            for (size_t j = 0; j < other.shape[1]; j++) {
                values.push_back(data[j] + other.data[i * other.shape[1] + j]);
            }
        }
        return Tensor(other.shape, values);
    }

    throw invalid_argument("Dimensiones incompatibles para suma");
}

// resta elemento a elemento
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape != other.shape) {
        throw invalid_argument("Dimensiones incompatibles");
    }

    vector<double> values;
    for (size_t i = 0; i < total_size; i++) {
        values.push_back(data[i] - other.data[i]);
    }

    return Tensor(shape, values);
}

// multiplicación elemento a elemento
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape != other.shape) {
        throw invalid_argument("Dimensiones incompatibles");
    }

    vector<double> values;
    for (size_t i = 0; i < total_size; i++) {
        values.push_back(data[i] * other.data[i]);
    }

    return Tensor(shape, values);
}

// multiplicación por escalar
Tensor Tensor::operator*(double scalar) const {
    vector<double> values;
    for (size_t i = 0; i < total_size; i++) {
        values.push_back(data[i] * scalar);
    }

    return Tensor(shape, values);
}

// imprime dimensiones
void Tensor::printShape() const {
    cout << "shape = {";
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i + 1 < shape.size()) cout << ", ";
    }
    cout << "}" << endl;
}

// impresión recursiva del tensor
void Tensor::printRecursive(size_t dimension, size_t& IDnumero) const {
    cout << "[";

    size_t contador = shape[dimension];

    if (dimension == shape.size() - 1) {
        //caso final
        for (size_t i = 0; i < contador; i++) {
            cout << data[IDnumero++];
            if (i + 1 < contador) cout << ", ";
        }

    } else {
        for (size_t i = 0; i < contador; i++) {
            //recursiva
            printRecursive(dimension + 1, IDnumero);
            if (i + 1 < contador) cout << ", ";
        }
    }

    cout << "]";
}

// inicia impresión
void Tensor::printData() const {
    size_t IDnumero = 0;
    printRecursive(0, IDnumero);
    cout << endl;
}

// producto punto
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw invalid_argument("Dimensiones incompatibles para dot");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.total_size; i++) {
        sum += a.data[i] * b.data[i];
    }

    return Tensor({1}, {sum});
}

// multiplicación matricial
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw invalid_argument("matmul solo acepta tensores 2D");
    }
    if (a.shape[1] != b.shape[0]) {
        throw invalid_argument("Dimensiones incompatibles para matmul");
    }

    size_t rows = a.shape[0];
    size_t inner = a.shape[1];
    size_t cols = b.shape[1];

    vector<double> values(rows * cols, 0.0);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < inner; k++) {
                sum += a.data[i * inner + k] * b.data[k * cols + j];
            }
            values[i * cols + j] = sum;
        }
    }

    return Tensor({rows, cols}, values);
}


