#include <iostream>
using namespace std;

const int N = 3; // Assuming 3x3 matrices for simplicity

// Function to multiply two matrices
void multiplyMatrices(int mat1[][N], int mat2[][N], int res[][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            res[i][j] = 0;
            for (int k = 0; k < N; k++) {
                res[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
}

// Function to print a matrix
void printMatrix(int mat[][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int mat1[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int mat2[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int res[N][N]; // To store result

    multiplyMatrices(mat1, mat2, res);

    cout << "Result of matrix multiplication: " << endl;
    printMatrix(res);

    return 0;
}
