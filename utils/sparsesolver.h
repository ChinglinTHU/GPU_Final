#pragma once

#include <cuda_runtime.h>
#include "cusparse.h"

class SparseSolver
{
public:
	SparseSolver();
	~SparseSolver();
	bool Solve(int* ARow, int* ACol, float* AVal, float* B, float* X, int m, int nnz);
	// void Result();

private:
	cusolverStatus_t solver_status;
	cusparseStatus_t sparse_status;
	cusolverSpHandle_t solver_handle;
	cusparseHandle_t sparse_handle;
	cusparseMatDescr_t descr;
	// int m, nnz;
	// int* csrRow;
	// int* csrCol;
	// float* csrVal;
	// float* b;
	// float* x;
};