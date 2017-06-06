#include "./sparsesolver.h"
#include <iostream>
#include <cuda_runtime.h>
#include "cusolverSp.h"

using namespace std;

bool cuSpSolver(int* ARow, int* ACol, float* AVal, float* B, float* X, int m, int nnz)
{
	cusolverStatus_t solver_status;
	cusparseStatus_t sparse_status;
	cusolverSpHandle_t solver_handle = 0;
	cusparseHandle_t sparse_handle = 0;
	cusparseMatDescr_t descr = 0;

	// initialize
	sparse_status = cusparseCreate(&sparse_handle);
	if(sparse_status != CUSPARSE_STATUS_SUCCESS)
	{
		cout << "ERROR: cusparseCreate failed" << endl;
		return 2;
	}
	sparse_status = cusparseCreateMatDescr(&descr);
	if(sparse_status != CUSPARSE_STATUS_SUCCESS)
	{
		cout << "ERROR: cusparseCreateMatDescr failed" << endl;
		return 2;
	}
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	solver_status = cusolverSpCreate(&solver_handle);
	if(solver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout << "ERROR: cusparseCreate failed" << endl;
		return 2;
	}

	int* csrRow = 0; cudaMalloc((void**)&csrRow, (m+1)*sizeof(csrRow[0]));
	int* cooRow = 0; cudaMalloc((void**)&cooRow, nnz*sizeof(cooRow[0]));
	int* csrCol = 0; cudaMalloc((void**)&csrCol, nnz*sizeof(csrCol[0]));
	float* csrVal = 0; cudaMalloc((void**)&csrVal, nnz*sizeof(csrVal[0]));
	float* b = 0; cudaMalloc((void**)&b, m*sizeof(b[0]));
	float* x = 0; cudaMalloc((void**)&x, m*sizeof(x[0]));

	cudaMemcpy(cooRow, ARow, (size_t)(nnz*sizeof(cooRow[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(csrCol, ACol, (size_t)(nnz*sizeof(csrCol[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(csrVal, AVal, (size_t)(nnz*sizeof(csrVal[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(b, B, (size_t)(m*sizeof(b[0])), cudaMemcpyHostToDevice);
	cusparseXcoo2csr(sparse_handle, cooRow, nnz, m, csrRow, CUSPARSE_INDEX_BASE_ZERO);

	int singular;
	solver_status = cusolverSpScsrlsvqr(
		solver_handle,
		m,
		nnz,
		descr,
		csrVal,
		csrRow,
		csrCol,
		b,
		0.0,
		0,
		x,
		&singular);
	if(solver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout<<"solve failed"<<endl;
		return 1;
	}
	else
	{
		cudaMemcpy(X, x, (size_t)(m*sizeof(X[0])), cudaMemcpyDeviceToHost);
	}

	cudaFree(csrRow);
	cudaFree(cooRow);
	cudaFree(csrVal);
	cudaFree(csrCol);
	cudaFree(x);
	cudaFree(b);
	cusolverSpDestroy(solver_handle);
	cusparseDestroy(sparse_handle);
	return 0;
}

// void Result();