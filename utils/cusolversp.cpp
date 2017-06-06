#include <iostream>
#include <cuda_runtime.h>
#include "cusolverSp.h"
#include "sparsesolver.cpp"
using namespace std;

int main()
{
	cudaError_t cudaStat;
	cusolverStatus_t solver_status;
	cusparseStatus_t sparse_status;
	cusolverSpHandle_t solver_handle = 0;
	cusparseHandle_t sparse_handle = 0;
	cusparseMatDescr_t descr = 0;

	int m = 4;
	int nnz = 9;

	// init host matrix
	int ARowHost[nnz] = {0,0,0,1,2,2,2,3,3};
	int AColHost[nnz] = {0,2,3,1,0,2,3,1,3};
	float AValHost[nnz] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
	// float x = {1.0, 1.0, 1.0, 1.0}
	float BHost[m] = {6.0, 4.0, 18.0, 17.0};
/*
	// init device matrix
	int* ARow = 0;  cudaMalloc((void**)&ARow, nnz*sizeof(ARow[0]));
	int* ACol = 0;  cudaMalloc((void**)&ACol, nnz*sizeof(ACol[0]));
	float* AVal = 0;cudaMalloc((void**)&AVal, nnz*sizeof(AVal[0]));
	float* B = 0;   cudaMalloc((void**)&B,    m*sizeof(B[0]));
	cudaMemcpy(ARow, ARowHost, (size_t)(nnz*sizeof(ARow[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(ACol, AColHost, (size_t)(nnz*sizeof(ACol[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(AVal, AValHost, (size_t)(nnz*sizeof(AVal[0])), cudaMemcpyHostToDevice);
	cudaMemcpy(B, BHost, (size_t)(m*sizeof(B[0])), cudaMemcpyHostToDevice);

	// create cusparse library
	solver_status = cusolverSpCreate(&solver_handle);
	if(solver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout<<"cusparseCreate failed"<<endl;
		return 1;
	}
	sparse_status = cusparseCreate(&sparse_handle);
	sparse_status = cusparseCreateMatDescr(&descr);
	if(sparse_status != CUSPARSE_STATUS_SUCCESS)
	{
		cout<<"cusparseCreateMatDescr failed"<<endl;
		return 1;
	}
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	// convert coo to csr
	int* csrRow = 0;
	cudaMalloc((void**)&csrRow, (m+1)*sizeof(csrRow[0]));
	cusparseXcoo2csr(sparse_handle, ARow, nnz, m, csrRow, CUSPARSE_INDEX_BASE_ZERO);

	// csr_solve
	float* x = 0;
	int singularity;
	cudaMalloc((void**)&x, m*sizeof(x[0]));
	solver_status = cusolverSpScsrlsvqr(
		solver_handle,
		m,
		nnz,
		descr,
		AVal,
		csrRow,
		ACol,
		B,
		0.0,
		0,
		x, 
		&singularity);
	if(solver_status != CUSOLVER_STATUS_SUCCESS)
	{
		cout<<"solve failed"<<endl;
		return 1;
	}
	else
		cout<<"solve success"<<endl;
*/
	float XHost[m] = {0.0};
	// cudaMemcpy(XHost, x, (size_t)(m*sizeof(x[0])), cudaMemcpyDeviceToHost);

	SparseSolver mySolver;
	mySolver.Solve(ARowHost, AColHost, AValHost, BHost, XHost, m, nnz);

	// output x
	cout << "x: [";
	for(int i = 0; i<m; i++)
	{
		cout<<XHost[i]<<" ";
	}
	cout << "]"<<endl;
	return 0;
}