#include "Tensor.h"
#include "TensorTransform.h"

int main() {
    try {
        cout << "=== Operaciones basicas ===" << endl;
        Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor B({2, 3}, {6, 5, 4, 3, 2, 1});

        Tensor C = A + B;
        Tensor D = A - B;
        Tensor E = A * B;
        Tensor F = A * 2.0;

        ReLU relu;
        Sigmoid sigmoid;
        Tensor G = A.apply(relu);


        C.printShape();
        C.printData();
        D.printData();
        E.printData();
        F.printData();
        G.printData();

        cout << "\n=== view, unsqueeze, concat, dot y matmul ===" << endl;
        Tensor V = Tensor::arange(0, 12);
        Tensor V2 = V.view({3, 4});
        V2.printShape();

        Tensor U = Tensor::arange(0, 3);
        Tensor U2 = U.unsqueeze(0);
        U2.printShape();

        Tensor C1 = Tensor::ones({2, 3});
        Tensor C2 = Tensor::zeros({2, 3});
        Tensor C3 = Tensor::concat({C1, C2}, 0);
        C3.printShape();

        Tensor DP = dot(C1, C1);
        DP.printData();

        Tensor M1({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor M2({3, 2}, {7, 8, 9, 10, 11, 12});
        Tensor MM = matmul(M1, M2);
        MM.printShape();
        MM.printData();

        cout << "\n=== Red neuronal ===" << endl;
        Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);
        Tensor flat = input.view({1000, 400});

        Tensor W1 = Tensor::random({400, 100}, -1.0, 1.0);
        Tensor b1 = Tensor::random({1, 100}, -1.0, 1.0);
        Tensor H1 = matmul(flat, W1) + b1;
        Tensor A1 = H1.apply(relu);

        Tensor W2 = Tensor::random({100, 10}, -1.0, 1.0);
        Tensor b2 = Tensor::random({1, 10}, -1.0, 1.0);
        Tensor H2 = matmul(A1, W2) + b2;
        Tensor output = H2.apply(sigmoid);

        output.printShape();
        cout << "Red neuronal ejecutada correctamente" << endl;
    }
    catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
    }

    return 0;
}
