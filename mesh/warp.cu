#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>

// #include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/opencv_modules.hpp"

#include "opencv2/stitching/detail/blenders.hpp"

//#include "opencv2/gpu.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

#include "warp.h"
#include "../utils/SyncedMemory.h"
#include "../utils/Timer.h"

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::cuda;

typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;
typedef vector<vector<Mat> > BundleHomo;
#define index(i, j) (j)*width+(i)
#define cellind(i, j) (j)*cellwidth+(i)

warp::warp(asapWarp asap)
{
	this->asap = asap;
	this->height = asap.height;
	this->width = asap.width;
	this->cellheight = this->height-1;
	this->cellwidth = this->width-1;

	vector<Point2f> cellPtsT(width*height);
	vector<Point2f> cellPts0(width*height);
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			cellPtsT[index(i, j)] = asap.compute_pos(i, j);
			cellPts0[index(i, j)] = asap.compute_pos(i, j);
		}

	this->cellPtsT = cellPtsT;
	this->cellPts0 = cellPts0;
}

warp::~warp(){}

__host__ __device__ int findCell(const float* point, const float x, const float y, const int pointNum, 
						const int cellwidth, const int cellheight)
{
	float minx, maxx, miny, maxy;
	minx = maxx = point[0];
	miny = maxy = point[1];
	int i, j;
	for (i = 1; i < pointNum; i++)
	{
		minx = point[i*2+0] < minx ? point[i*2+0] : minx;
		maxx = point[i*2+0] > maxx ? point[i*2+0] : maxx;
		miny = point[i*2+1] < miny ? point[i*2+1] : miny;
		maxy = point[i*2+1] > maxy ? point[i*2+1] : maxy;
	}

	if (x > maxx || x < minx || y > maxy || y < miny)
		return -1;
	
	float V00x, V00y, V10x, V10y, V01x, V01y, V11x, V11y;
	float ax, ay, bx, by;
	for (i = 0; i < cellwidth; i++)
		for (j = 0; j < cellheight; j++)
		{

			V00x = point[2*(j*(cellwidth+1)+i)+0];
			V00y = point[2*(j*(cellwidth+1)+i)+1];
			V10x = point[2*(j*(cellwidth+1)+i+1)+0];
			V10y = point[2*(j*(cellwidth+1)+i+1)+1];
			V01x = point[2*((j+1)*(cellwidth+1)+i)+0];
			V01y = point[2*((j+1)*(cellwidth+1)+i)+1];
			V11x = point[2*((j+1)*(cellwidth+1)+i+1)+0];
			V11y = point[2*((j+1)*(cellwidth+1)+i+1)+1];

			minx = V00x < V01x ? V00x : V01x;
			maxx = V10x > V11x ? V10x : V11x;
			miny = V00y < V10y ? V00y : V10y;
			maxy = V01y > V11y ? V01y : V11y;

			if (x > maxx || x < minx || y > maxy || y < miny)
				continue;
		
			ax = V10x-V00x;
			ay = V10y-V00y;
			bx = x-V00x;
			by = y-V00y;
			if (ax*by-ay*bx < 0)
				continue;

			ax = V11x-V10x;
			ay = V11y-V10y;
			bx = x-V10x;
			by = y-V10y;
			if (ax*by-ay*bx < 0)
				continue;

			ax = V01x-V11x;
			ay = V01y-V11y;
			bx = x-V11x;
			by = y-V11y;
			if (ax*by-ay*bx < 0)
				continue;

			ax = V00x-V01x;
			ay = V00y-V01y;
			bx = x-V01x;
			by = y-V01y;
			if (ax*by-ay*bx < 0)
				continue;
			
			return j*cellwidth+i;
		}

	return -1;
}

__global__ void warpImgByVertexGPU(PtrStepSz<uchar3> const img, PtrStepSz<uchar3> warpimg, 
						const float* ptT, const float* warppt0, const float* Phinv, const float* CH, 
						const int cellwidth, const int cellheight, const int pointNum)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float shared_ptT[192];
	__shared__ float shared_warppt0[192];
	int idx;
	for (int i = 0; i < pointNum / (blockDim.x * blockDim.y) + 1; i++)
	{
		idx = 2*((blockDim.x * blockDim.y)*i + threadIdx.y * blockDim.x + threadIdx.x);
		if (idx < pointNum*2)
		{
			shared_ptT[idx] = ptT[idx];
			shared_ptT[idx+1] = ptT[idx+1];
			shared_warppt0[idx] = warppt0[idx];
			shared_warppt0[idx+1] = warppt0[idx+1];
		}	
	}

	__shared__ float shared_Phinv[576];
	__shared__ float shared_CH[576];
	idx = threadIdx.y * blockDim.x + threadIdx.x;
	if (idx < 576)
	{
		shared_Phinv[idx] = Phinv[idx];
		shared_CH[idx] = CH[idx];
	}

	__syncthreads();

	if (x < img.cols && y <= img.rows)
	{
		float ptx = float(x);
		float pty = float(y);
		int cellindex = findCell(shared_ptT, ptx, pty, pointNum, cellwidth, cellheight);

		if (cellindex < 0)
			return;

		float warpTx = shared_Phinv[9*cellindex+0]*ptx + shared_Phinv[9*cellindex+1]*pty + shared_Phinv[9*cellindex+2]*1.f;
		float warpTy = shared_Phinv[9*cellindex+3]*ptx + shared_Phinv[9*cellindex+4]*pty + shared_Phinv[9*cellindex+5]*1.f;
		float warpTz = shared_Phinv[9*cellindex+6]*ptx + shared_Phinv[9*cellindex+7]*pty + shared_Phinv[9*cellindex+8]*1.f;
		warpTx = warpTx / warpTz;
		warpTy = warpTy / warpTz;

		cellindex = findCell(shared_warppt0, warpTx, warpTy, pointNum, cellwidth, cellheight);

		if (cellindex < 0)
			return;
		float warpTx2 = shared_CH[9*cellindex+0]*warpTx + shared_CH[9*cellindex+1]*warpTy + shared_CH[9*cellindex+2]*1.f;
		float warpTy2 = shared_CH[9*cellindex+3]*warpTx + shared_CH[9*cellindex+4]*warpTy + shared_CH[9*cellindex+5]*1.f;
		float warpTz2 = shared_CH[9*cellindex+6]*warpTx + shared_CH[9*cellindex+7]*warpTy + shared_CH[9*cellindex+8]*1.f;
		warpTx2 = warpTx2 / warpTz2;
		warpTy2 = warpTy2 / warpTz2;

		int floorx = int(warpTx2);
		int floory = int(warpTy2);
		float deltax = warpTx2-float(floorx);
		float deltay = warpTy2-float(floory);

		if (floorx > img.cols-1 || floorx+1 < 0 || floory > img.rows-1 || floory+1 < 0)
			return;
		
		uchar3 color00 = img(floory, floorx);
		uchar3 color10 = img(floory+1, floorx);
		uchar3 color01 = img(floory, floorx+1);
		uchar3 color11 = img(floory+1, floorx+1);

		float b = (1-deltax)*(1-deltay)*float(color00.x) 
				+ (1-deltax)*(deltay)  *float(color10.x)
				+ (deltax)*(deltay)    *float(color11.x)
				+ (deltax)*(1-deltay)  *float(color01.x);
		float g = (1-deltax)*(1-deltay)*float(color00.y) 
				+ (1-deltax)*(deltay)  *float(color10.y)
				+ (deltax)*(deltay)    *float(color11.y)
				+ (deltax)*(1-deltay)  *float(color01.y);
		float r = (1-deltax)*(1-deltay)*float(color00.z) 
				+ (1-deltax)*(deltay)  *float(color10.z)
				+ (deltax)*(deltay)    *float(color11.z)
				+ (deltax)*(1-deltay)  *float(color01.z);
		b = b > 255 ? 255 : b;
		b = b < 0 ? 0 : b;
		g = g > 255 ? 255 : g;
		g = g < 0 ? 0 : g;
		r = r > 255 ? 255 : r;
		r = r < 0 ? 0 : r;
		
		warpimg(y, x).x = uchar(b);
		warpimg(y, x).y = uchar(g);
		warpimg(y, x).z = uchar(r);
		
	}
}

vector<Point2i> warp::compute_corner(vector<Point2f> warpPts0, float *Pinv)
{
	vector<Point2i> corner;
	Point2f V00 = warpPts0[index(0, 0)];
	Point2f V01 = warpPts0[index(0, height-1)];
	Point2f V10 = warpPts0[index(width-1, 0)];
	Point2f V11 = warpPts0[index(width-1, height-1)];

	Mat Pinv00(3, 3, CV_32F);
	Mat Pinv01(3, 3, CV_32F);
	Mat Pinv10(3, 3, CV_32F);
	Mat Pinv11(3, 3, CV_32F);	

	int ind = cellind(0, 0);
	Pinv00.at<float>(0, 0) = Pinv[9*ind+0];
	Pinv00.at<float>(0, 1) = Pinv[9*ind+1];
	Pinv00.at<float>(0, 2) = Pinv[9*ind+2];
	Pinv00.at<float>(1, 0) = Pinv[9*ind+3];
	Pinv00.at<float>(1, 1) = Pinv[9*ind+4];
	Pinv00.at<float>(1, 2) = Pinv[9*ind+5];
	Pinv00.at<float>(2, 0) = Pinv[9*ind+6];
	Pinv00.at<float>(2, 1) = Pinv[9*ind+7];
	Pinv00.at<float>(2, 2) = Pinv[9*ind+8];

	ind = cellind(cellwidth-1, 0);
	Pinv10.at<float>(0, 0) = Pinv[9*ind+0];
	Pinv10.at<float>(0, 1) = Pinv[9*ind+1];
	Pinv10.at<float>(0, 2) = Pinv[9*ind+2];
	Pinv10.at<float>(1, 0) = Pinv[9*ind+3];
	Pinv10.at<float>(1, 1) = Pinv[9*ind+4];
	Pinv10.at<float>(1, 2) = Pinv[9*ind+5];
	Pinv10.at<float>(2, 0) = Pinv[9*ind+6];
	Pinv10.at<float>(2, 1) = Pinv[9*ind+7];
	Pinv10.at<float>(2, 2) = Pinv[9*ind+8];

	ind = cellind(0, cellheight-1);
	Pinv01.at<float>(0, 0) = Pinv[9*ind+0];
	Pinv01.at<float>(0, 1) = Pinv[9*ind+1];
	Pinv01.at<float>(0, 2) = Pinv[9*ind+2];
	Pinv01.at<float>(1, 0) = Pinv[9*ind+3];
	Pinv01.at<float>(1, 1) = Pinv[9*ind+4];
	Pinv01.at<float>(1, 2) = Pinv[9*ind+5];
	Pinv01.at<float>(2, 0) = Pinv[9*ind+6];
	Pinv01.at<float>(2, 1) = Pinv[9*ind+7];
	Pinv01.at<float>(2, 2) = Pinv[9*ind+8];

	ind = cellind(cellwidth-1, cellheight-1);
	Pinv11.at<float>(0, 0) = Pinv[9*ind+0];
	Pinv11.at<float>(0, 1) = Pinv[9*ind+1];
	Pinv11.at<float>(0, 2) = Pinv[9*ind+2];
	Pinv11.at<float>(1, 0) = Pinv[9*ind+3];
	Pinv11.at<float>(1, 1) = Pinv[9*ind+4];
	Pinv11.at<float>(1, 2) = Pinv[9*ind+5];
	Pinv11.at<float>(2, 0) = Pinv[9*ind+6];
	Pinv11.at<float>(2, 1) = Pinv[9*ind+7];
	Pinv11.at<float>(2, 2) = Pinv[9*ind+8];

	vector<Point2f> pt(1);
	vector<Point2f> warppt(1);
	pt[0] = V00;
	perspectiveTransform(pt, warppt, Pinv00.inv());
	corner.push_back(Point2i(warppt[0]));

	pt[0] = V01;
	perspectiveTransform(pt, warppt, Pinv01.inv());
	corner.push_back(Point2i(warppt[0]));

	pt[0] = V10;
	perspectiveTransform(pt, warppt, Pinv10.inv());
	corner.push_back(Point2i(warppt[0]));

	pt[0] = V11;
	perspectiveTransform(pt, warppt, Pinv11.inv());
	corner.push_back(Point2i(warppt[0]));

	return corner;
}

void warp::findCut(Mat img, int* cutxy, vector<Point2i> corner)
{
	int minx = min(max(max(corner[0].x, corner[1].x), 0), img.cols/3);
	int miny = min(max(max(corner[0].y, corner[2].y), 0), img.rows/3);
	int maxx = max(min(min(corner[2].x, corner[3].x), img.cols-1), 2*img.cols/3);
	int maxy = max(min(min(corner[1].y, corner[3].y), img.rows-1), 2*img.rows/3);
	uchar zero = uchar(0);

	for (int x = minx; x <= maxx; x++)
	{
		for (int y = miny; y < miny+10; y++)
		{
			//printf("img.at<uchar3>(%d, %d) = (%u, %u, %u)\n", y, x, img.at<uchar3>(y, x).x, img.at<uchar3>(y, x).y, img.at<uchar3>(y, x).z);
			if (img.at<uchar3>(y, x).x != zero || img.at<uchar3>(y, x).y != zero || img.at<uchar3>(y, x).z != zero)
			{
				if (miny < y)
					miny = y;
				break;
			}
		}
	}
	for (int x = minx; x <= maxx; x++)
	{
		for (int y = maxy; y > maxy-10; y--)
		{
			//printf("img.at<uchar3>(%d, %d) = (%u, %u, %u)\n", y, x, img.at<uchar3>(y, x).x, img.at<uchar3>(y, x).y, img.at<uchar3>(y, x).z);
			if (img.at<uchar3>(y, x).x != zero || img.at<uchar3>(y, x).y != zero || img.at<uchar3>(y, x).z != zero)
			{
				
				if (maxy > y)
					maxy = y;
				break;
			}
		}
	}

	for (int y = miny; y <= maxy; y++)
	{
		for (int x = minx; x < minx+10; x++)
		{
			if (img.at<uchar3>(y, x).x != zero || img.at<uchar3>(y, x).y != zero || img.at<uchar3>(y, x).z != zero)
			{
				if (minx < x)
					minx = x;
				break;
			}
		}
	}
	for (int y = miny; y <= maxy; y++)
	{
		for (int x = maxx; x > maxx-10; x--)
		{
			if (img.at<uchar3>(y, x).x != zero || img.at<uchar3>(y, x).y != zero || img.at<uchar3>(y, x).z != zero)
			{
				if (maxx > x)
					maxx = x;
				break;
			}
		}
	}

	cutxy[0] = minx;
	cutxy[1] = maxx;
	cutxy[2] = miny;
	cutxy[3] = maxy;

	/*
	cout << "minx = " << minx << endl;
	cout << "maxx = " << maxx << endl;
	cout << "miny = " << miny << endl;
	cout << "maxy = " << maxy << endl;
	//*/

	// resize(img(Rect(minx, miny, sizex, sizey)), cutimg, img.size());
	return;
}

void warp::compute_homo(float *C, const vector<Point2f> &pts, const vector<Point2f> &warpPts)
{
	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
		{
			int ind = 9*(j*(width-1)+i);
			vector<Point2f> P, WP;
			for (int I = i; I < i+2; I++)
				for (int J = j; J < j+2; J++)
				{
					P.push_back(pts[index(I, J)]);
					WP.push_back(warpPts[index(I, J)]);
				}

			Mat h = findHomography(P, WP);
			C[ind+0] = float(h.at<double>(0, 0));
			C[ind+1] = float(h.at<double>(0, 1));
			C[ind+2] = float(h.at<double>(0, 2));
			C[ind+3] = float(h.at<double>(1, 0));
			C[ind+4] = float(h.at<double>(1, 1));
			C[ind+5] = float(h.at<double>(1, 2));
			C[ind+6] = float(h.at<double>(2, 0));
			C[ind+7] = float(h.at<double>(2, 1));
			C[ind+8] = float(h.at<double>(2, 2));
		}
}

void warp::warpImageMeshbyVertexGPU(Mat img, Mat & warpimg, vector<Point2f> warpPts0, vector<Point2f> warpPtsT, int * cutxy)
{
	float *Pinv = new float[(width-1)*(height-1)*9];
 	compute_homo(Pinv, cellPtsT, warpPtsT);
	float *C = new float[(width-1)*(height-1)*9];
	compute_homo(C, warpPts0, cellPts0);

	int N = cellPtsT.size();
	float *ptrT  = new float[N*2];
	float *warp0 = new float[N*2];
	for (int i = 0; i < N; i++)
	{
		ptrT[i*2+0] = cellPtsT[i].x;
		ptrT[i*2+1] = cellPtsT[i].y;
		warp0[i*2+0] = warpPts0[i].x;
		warp0[i*2+1] = warpPts0[i].y;
	}
	
	float *ptrT_mat, *warp0_mat;
	cudaMalloc((void **) &ptrT_mat,   N*2*sizeof(float));
	cudaMalloc((void **) &warp0_mat, N*2*sizeof(float));
	cudaMemcpy(ptrT_mat,   ptrT,  N*2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(warp0_mat, warp0, N*2*sizeof(float), cudaMemcpyHostToDevice);

	int M = (width-1)*(height-1);
	float *Pinv_device, *C_device;
	cudaMalloc((void **) &Pinv_device, M*9*sizeof(float));
	cudaMalloc((void **) &C_device,    M*9*sizeof(float));
	cudaMemcpy(Pinv_device, Pinv,  M*9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_device,    C,     M*9*sizeof(float), cudaMemcpyHostToDevice);

	const int BLOCK_WIDTH = 32;
	const int BLOCK_HEIGHT = 32;
	const int X_BLOCK_NUM = (img.cols-1)/BLOCK_WIDTH + 1;
	const int Y_BLOCK_NUM = (img.rows-1)/BLOCK_HEIGHT + 1;
	dim3 block = dim3(X_BLOCK_NUM, Y_BLOCK_NUM);
	dim3 thread = dim3(BLOCK_WIDTH, BLOCK_HEIGHT);

	GpuMat img_device; 
	img_device.upload(img);
	GpuMat warpimg_device(img.size(), CV_8UC3);
	warpimg_device.setTo(0);


	//warpImgByVertexGPU(PtrStepSz<uchar3> const img, PtrStepSz<uchar3> warpimg, 
	//					const float* pt, const float* warppt0, const float* PHinv, const float* C,
	//					const int cellwidth, const int cellheight, const int pointNum)
	if (N != 81)
	{
		cerr << "pointNum must equal to the given number 81 in the cuda code, need to modify the cuda code" << endl;
		return;
	}

	warpImgByVertexGPU<<< block, thread >>>(img_device, warpimg_device, ptrT_mat, warp0_mat, 
											Pinv_device, C_device, width-1, height-1, N);
	warpimg_device.download(warpimg);

	// minx, maxx, miny, maxy

	// void findCut(float *cutxy, float* warpT, float *warp0, float *Pinv, float *C, const int cellwidth, const int cellheight, const int pointNum)
	vector<Point2i> corner = compute_corner(warpPts0, Pinv);
	findCut(warpimg, cutxy, corner);
	for (int i = 0; i < 4; i++)
		cout << "corner = " << corner[i] << endl;

	cout << "cutx = " << cutxy[0] << ", " << cutxy[1] << endl;
	cout << "cuty = " << cutxy[2] << ", " << cutxy[3] << endl;

	/* imshow
	namedWindow("warpimg", WINDOW_AUTOSIZE);
	imshow("warpimg", warpimg);
	waitKey(0);
	//*/

	return;
}

void warp::warpImageMeshGPU(Mat img, Mat & warpimg, BundleHomo Cpath, BundleHomo P)
{
	vector<Point2f> cellPtsT(width*height);
	vector<Point2f> cellPts0(width*height);
	vector<Point2f> warpPtsT(width*height);
	vector<Point2f> warpPts0(width*height);
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			cellPtsT[index(i, j)] = asap.compute_pos(i, j);
			cellPts0[index(i, j)] = asap.compute_pos(i, j);
		}

	vector<Point2f> pt(1);
	vector<Point2f> warpPt(1);

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPts0[index(i, j)];
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, Cpath[I][J].inv());
					sumpt += warpPt[0];
				}

			warpPts0[index(i, j)] = sumpt / N;
 		}

 	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPtsT[index(i, j)];
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, P[I][J].inv());
					sumpt += warpPt[0];
				}

			warpPtsT[index(i, j)] = sumpt / N;		
		}

	float *Pinv = new float[(width-1)*(height-1)*9];
 	compute_homo(Pinv, cellPtsT, warpPtsT);
	float *C = new float[(width-1)*(height-1)*9];
	compute_homo(C, warpPts0, cellPts0);

	int N = cellPtsT.size();
	float *ptrT  = new float[N*2];
	float *warp0 = new float[N*2];
	for (int i = 0; i < N; i++)
	{
		ptrT[i*2+0] = cellPtsT[i].x;
		ptrT[i*2+1] = cellPtsT[i].y;
		warp0[i*2+0] = warpPts0[i].x;
		warp0[i*2+1] = warpPts0[i].y;
	}
	
	float *ptrT_mat, *warp0_mat;
	cudaMalloc((void **) &ptrT_mat,   N*2*sizeof(float));
	cudaMalloc((void **) &warp0_mat, N*2*sizeof(float));
	cudaMemcpy(ptrT_mat,   ptrT,  N*2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(warp0_mat, warp0, N*2*sizeof(float), cudaMemcpyHostToDevice);

	int M = (width-1)*(height-1);
	float *Pinv_device, *C_device;
	cudaMalloc((void **) &Pinv_device, M*9*sizeof(float));
	cudaMalloc((void **) &C_device,    M*9*sizeof(float));
	cudaMemcpy(Pinv_device, Pinv,  M*9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(C_device,    C,     M*9*sizeof(float), cudaMemcpyHostToDevice);


	const int BLOCK_WIDTH = 16;
	const int BLOCK_HEIGHT = 12;
	const int X_BLOCK_NUM = (img.cols-1)/BLOCK_WIDTH + 1;
	const int Y_BLOCK_NUM = (img.rows-1)/BLOCK_HEIGHT + 1;
	dim3 block = dim3(X_BLOCK_NUM, Y_BLOCK_NUM);
	dim3 thread = dim3(BLOCK_WIDTH, BLOCK_HEIGHT);

	GpuMat img_device; 
	img_device.upload(img);
	GpuMat warpimg_device(img.size(), CV_8UC3);
	warpimg_device.setTo(0);


	//warpImgByVertexGPU(PtrStepSz<uchar3> const img, PtrStepSz<uchar3> warpimg, 
	//					const float* pt, const float* warppt0, const float* PHinv, const float* C,
	//					const int cellwidth, const int cellheight, const int pointNum)
	warpImgByVertexGPU<<< block, thread >>>(img_device, warpimg_device, ptrT_mat, warp0_mat, 
											Pinv_device, C_device, width-1, height-1, N);
	warpimg_device.download(warpimg);

	/* imshow
	namedWindow("warpimg", WINDOW_AUTOSIZE);
	imshow("warpimg", warpimg);
	waitKey(0);
	//*/

	return;
}

void warp::warpImageMesh(Mat img, Mat & warpimg, BundleHomo C, BundleHomo P)
{
	vector<Point2f> cellPtsT(width*height);
	vector<Point2f> cellPts0(width*height);
	
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			cellPtsT[index(i, j)] = asap.compute_pos(i, j);
		}

	vector<Point2f> pt(1);
	vector<Point2f> warpPt(1);

	float minx, miny, maxx, maxy;
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPtsT[index(i, j)];
//cerr << "(" << i << ", " << j << ")" << endl;
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, C[I][J].inv());
//cerr << "\t(" << I << ", " << J << ") " << endl;
//cerr << "C[I][J].inv() = " << C[I][J].inv() << endl;
//cerr << "warpPt = " << warpPt[0] << endl;
					sumpt += warpPt[0];
				}

			cellPts0[index(i, j)] = sumpt / N;
//cerr << "(" << i << ", " << j << ") = " << cellPts0[index(i, j)] << endl;
			if (i == 0 && j == 0)
			{
				minx = cellPts0[index(i, j)].x;
				maxx = cellPts0[index(i, j)].x;
				miny = cellPts0[index(i, j)].y;
				maxy = cellPts0[index(i, j)].y;
			}
			else
			{
				minx = min(cellPts0[index(i, j)].x, minx);
				maxx = max(cellPts0[index(i, j)].x, maxx);
				miny = min(cellPts0[index(i, j)].y, miny);
				maxy = max(cellPts0[index(i, j)].y, maxy);
			}

 		}

	float dist = max(maxx-minx, maxy-miny); 
	float rate = 1.f;
cerr << "dist = " << dist << endl;
	if (dist > 4000.f)
	{
		rate = 4000.f/dist;
		for (int i = 0; i < cellPts0.size(); i++)
			cellPts0[i] = cellPts0[i] * rate;
	}

	Mat originImg;
	Point2f offset = warpImgByVertex(img, originImg, cellPtsT, cellPts0, true);
	/* imshow
	namedWindow("originImg", WINDOW_AUTOSIZE);
	imshow("originImg", originImg);
	waitKey(0);
	//*/

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			int N = 0;
			Point2f sumpt(0.f, 0.f);
			pt[0] = cellPtsT[index(i, j)];
			for (int I = max(0, i-1); I < min(width-1, i+1); I++)
				for (int J = max(0, j-1); J < min(height-1, j+1); J++)
				{
					N++;
					perspectiveTransform(pt, warpPt, P[I][J].inv());
					sumpt += warpPt[0];
				}

			cellPts0[index(i, j)] = sumpt / N;
////cerr << "(" << i << ", " << j << ") = " << cellPts0[index(i, j)] << endl;			
		}

	if (dist > 4000.f)
	{
		rate = 4000.f/dist;
		for (int i = 0; i < cellPts0.size(); i++)
			cellPts0[i] = cellPts0[i] * rate;
	}

	warpImgByVertex(originImg, warpimg, cellPts0, cellPtsT, false, offset, img.size());
	originImg.release();

	/* imshow
	namedWindow("warpimg", WINDOW_AUTOSIZE);
	imshow("warpimg", warpimg);
	waitKey(0);
	//*/

}


Point warp::warpImgByVertex(Mat img, Mat & warpimg, vector<Point2f> pt, vector<Point2f> warppt, 
						bool all, Point offset, Size s)
{
	// offset means that offset -> (0, 0) 

	int minx = img.size().width;
	int maxx = 0;
	int miny = img.size().height;
	int maxy = 0;
	for (int i = 0; i < pt.size(); i++)
	{
		pt[i].x += offset.x;
		pt[i].y += offset.y;
		warppt[i].x += offset.x;
		warppt[i].y += offset.y;

		minx = min(int(floor(warppt[i].x)), minx);
		miny = min(int(floor(warppt[i].y)), miny);
		maxx = max(int(ceil(warppt[i].x)), maxx);
		maxy = max(int(ceil(warppt[i].y)), maxy);
	}
	offset.x -= minx;
	offset.y -= miny;

	for (int i = 0; i < pt.size(); i++)
	{
		warppt[i].x -= minx;
		warppt[i].y -= miny;
	}
	maxx -= minx;
	minx -= minx;
	maxy -= miny;
	miny -= miny;
	
	int sizex = maxx - minx + 1;
	int sizey = maxy - miny + 1;

	FeatherBlender blender(0.5f);  //sharpness
	blender.prepare(Rect(0, 0, sizex, sizey));

	Mat frame;
	img.convertTo(frame, CV_16SC3);

	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
		{
			Mat warp_frame, mask, h;
			mask = Mat::zeros(Size(sizex, sizey), CV_8U);

			Point countour[1][4];
			countour[0][0] = Point(warppt[index(i, j)]);
			countour[0][1] = Point(warppt[index(i, j+1)]);
			countour[0][2] = Point(warppt[index(i+1, j+1)]);
			countour[0][3] = Point(warppt[index(i+1, j)]);			

			const Point* ppt[1] = { countour[0] }; 
			int npt[] = {4};  
			fillPoly(mask, ppt, npt, 1, Scalar::all(255)); 
			polylines(mask, ppt, npt, 1, 1, Scalar::all(255), 10);  

			vector<Point2f> P, WP;
			for (int I = i; I < i+2; I++)
				for (int J = j; J < j+2; J++)
				{
					P.push_back(pt[index(I, J)]);
					WP.push_back(warppt[index(I, J)]);
				}

			h = findHomography(P, WP);
////cerr << "(" << i << ", " << j << ") = " << h << endl;	
			warpPerspective(frame, warp_frame, h, mask.size());
			// fb->feed(warp_frame, mask, Point(0, 0));
			blender.feed(warp_frame, mask, Point(0, 0));
		}

	Mat mask = Mat::zeros(Size(sizex, sizey), CV_8U);
	if (!all)
	{
		minx = offset.x;
		maxx = min(offset.x + s.width - 1, s.width- 1);
		miny = offset.y;
		maxy = min(offset.y + s.height - 1, s.height - 1);
	}
	else
	{
		s.width = sizex;
		s.height = sizey;
	}
	sizex = maxx - minx + 1;
	sizey = maxy - miny + 1;

	Mat warp_frame;

	mask(Rect(minx, miny, sizex, sizey)).setTo(Scalar::all(255));
	// fb->blend(warp_frame, mask);
	blender.blend(warp_frame, mask);
	//free(fb);
	warp_frame.convertTo(warp_frame, CV_8UC3);

	///* Draw Points on warpframe
	Mat warp_frame_points;
	DrawPoints(warp_frame, warp_frame_points, warppt, Point(0, 0));
	namedWindow("warp_frame_points", WINDOW_NORMAL);
	imshow("warp_frame_points", warp_frame_points);
	waitKey(0);
	//*/

	warpimg = Mat::zeros(s, warp_frame.type());
	warp_frame(Rect(minx, miny, sizex, sizey)).copyTo(warpimg(Rect(0, 0, sizex, sizey)));

	frame.release();
	warp_frame.release();
	mask.release();

	return offset;
}

void warp::DrawPoints(Mat img, Mat & pointImg, vector<Point2f> pts, Point offset)
{
	pointImg = img.clone();
	for (int i = 0; i < pts.size(); i++)
	{
		circle(pointImg, Point(pts[i])+offset, 3, Scalar(0, 0, 255), -1);
	}
}

#undef index
#undef cellind