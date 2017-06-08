#include "allPath.h"

#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cv::cuda;

typedef vector<Mat> Path;

#define PI 3.1415926f
#define gaussian(x, mean, var) exp(-((x-mean)/var)*((x-mean)/var)/2.f)/(var*sqrt(2*PI))

allPath::allPath(int height, int width, int t)
{

	this->height = height;
	this->width = width;
	this->time = t;

	vector<vector<Path> > cellPath(width, vector<Path> (height));
	vector<vector<Path> > cellHomo(width, vector<Path> (height));
	vector<vector<Path> > warpHomo(width, vector<Path> (height));
	vector<vector<Path> > optPath(width, vector<Path> (height));
	vector<vector<Path> > tmpPath(width, vector<Path> (height));

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			warpHomo[i][j] = vector<Mat>(this->time);
			optPath[i][j] = vector<Mat>(this->time);
			tmpPath[i][j] = vector<Mat>(this->time);
			cellPath[i][j] = vector<Mat>(this->time);
			cellHomo[i][j] = vector<Mat>(this->time-1);
			
			for (int t = 0; t < this->time; t++)
			{
				warpHomo[i][j][t] = Mat::eye(3, 3, CV_32FC1);
				cellPath[i][j][t] = Mat::eye(3, 3, CV_32FC1);
				optPath[i][j][t] = Mat::eye(3, 3, CV_32FC1);
				tmpPath[i][j][t] = Mat::eye(3, 3, CV_32FC1);
				if (t < this->time-1)
				{
					cellHomo[i][j][t] = Mat::eye(3, 3, CV_32FC1);	
				}
			}
		}

	this->cellPath = cellPath;
	this->cellHomo = cellHomo;
	this->warpHomo = warpHomo;
	this->optPath = optPath;
	this->tmpPath = tmpPath;
}

allPath::~allPath(){}

void allPath::setHomo(int i, int j, int t, Vec9f p)
{
	if (i >= width || j >= height || t >= time-1 || i < 0 || j < 0 || t < 0)
		throw runtime_error("allPath::setHomo: index can only inside the cell.\n");
	cellHomo[i][j][t].at<float>(0, 0) = p[0];
	cellHomo[i][j][t].at<float>(0, 1) = p[1];
	cellHomo[i][j][t].at<float>(0, 2) = p[2];
	cellHomo[i][j][t].at<float>(1, 0) = p[3];
	cellHomo[i][j][t].at<float>(1, 1) = p[4];
	cellHomo[i][j][t].at<float>(1, 2) = p[5];
	cellHomo[i][j][t].at<float>(2, 0) = p[6];
	cellHomo[i][j][t].at<float>(2, 1) = p[7];
	cellHomo[i][j][t].at<float>(2, 2) = p[8];
}


void allPath::setHomo(int i, int j, int t, Mat h)
{
	if (i >= width || j >= height || t >= time-1 || i < 0 || j < 0 || t < 0)
		throw runtime_error("allPath::setHomo: index can only inside the cell.\n");
	cellHomo[i][j][t] = h;
}

void allPath::setPath(int i, int j, Path p)
{
	if (i >= width || j >= height || i < 0 || j < 0)
		throw runtime_error("allPath::setPath: index can only inside the cell.\n");
	if (p.size() < time)
		throw runtime_error("allPath::setPath: p time must big enough.\n");
	cellPath[i][j] = p;
}

void allPath::computePath()
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			cellPath[i][j][0] = Mat::eye(3, 3, CV_32FC1);
			for (int t = 1; t < time; t++)
			{
				cellPath[i][j][t] = cellPath[i][j][t-1] * cellHomo[i][j][t-1];
				optPath[i][j][t] = cellPath[i][j][t];
				tmpPath[i][j][t] = cellPath[i][j][t];
			}
		}

}

void allPath::computeWarp()
{
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			for (int t = 0; t < time; t++)
				warpHomo[i][j][t] = cellPath[i][j][t].inv(DECOMP_SVD)*optPath[i][j][t];
		

}

Mat allPath::getWarpHomo(int i, int j, int t)
{
	if (i >= width || j >= height || t >= time || i < 0 || j < 0 || t < 0)
		throw runtime_error("allPath::getWarpHomo: index can only inside the cell.\n");
	return warpHomo[i][j][t];
}

Path allPath::getPath(int i, int j)
{
	
}

Path allPath::getOptimizedPath(int i, int j)
{
	if (i >= width || j >= height || i < 0 || j < 0)
		throw runtime_error("allPath::getOptimizedPath: index can only inside the cell.\n");
	return optPath[i][j];
}

void allPath::optimizePath(int iter)
{
	float w = .3f;

	printf("Optimizing path: \n");
	for (int k = 0; k < iter; k++)
	{
		printf("\titer %d \n", k);
		int sta_t, end_t;
		for (int t = 0; t < time; t++)
		{
			sta_t = t-30 < 0 ? 0 : t-30;
			end_t = t+30 > time-1 ? time-1 : t+30;

			int sta_i, end_i, sta_j, end_j;
			for (int i = 0; i < width; i++)
				for (int j = 0; j < height; j++)
				{
					sta_i = i-1 < 0 ? 0 : i-1;
					end_i = i+1 > width-1 ? width-1 : i+1;
					sta_j = j-1 < 0 ? 0 : j-1;
					end_j = j+1 > width-1 ? width-1 : j+1;
					
					int num = 0;
					float weight = 0.f;
					Mat sum = Mat::zeros(3, 3, CV_32FC1);
					for (int T = sta_t; T <= end_t; T++)
						for (int I = sta_i; I <= end_i; I++)
							for (int J = sta_j; J <= end_j; J++)
							{
								// cout << gaussian(T, t, 10)*gaussian(I, i, 1)*gaussian(J, j, 1) << "\n";
								weight += gaussian(float(T), float(t), 10.f)*gaussian(float(I), float(i), 1)*gaussian(float(J), float(j), 1);
								sum += gaussian(float(T), float(t), 10)*gaussian(float(I), float(i), 1)*gaussian(float(J), float(j), 1)*tmpPath[I][J][T];
							}

					sum += w*cellPath[i][j][t];
					weight += w;
					sum /= weight;
					optPath[i][j][t] = sum;
					tmpPath[i][j][t] = optPath[i][j][t];
				}
			
		}
	}
}
    