#pragma once

bool cuSpSolver(int* ARow, int* ACol, float* AVal, float* B, float* X, int m, int nnz);
bool cuSpSolver(int* ARow, int* ACol, float* AVal, float* B, float* X, int m, int n, int nnz);