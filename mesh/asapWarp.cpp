#include "asapWarp.h"

#include <cmath>

using namespace std;
using namespace cv;

asapWarp::asapWarp(int height, int width, int quadWidth, int quadHeight, float weight)
{
	source = Mesh(height, width, quadWidth, quadHeight);
	destin = Mesh(height, width, quadWidth, quadHeight);

	imgHeight = height;
	imgWidth = width;
	this.quadWidth = quadWidth;
	this.quadHeight = quadHeight;
	this.height = source.meshHeight;
	this.width = source.meshWidth;

	int tmp = this.height * this.width;
	for(int i = 0; i < tmp; i++)
	{
		x_index.push_back(i);
		y_index.push_back(tmp*2+i);
	}

	num_smooth_cons = (this.height-2)*(this.width-2)*16 + (2*(this.width+this.height)-8)*8+4*4;
	columns = tmp*2;

	SmoothConstraints = Mat::zeros(num_smooth_cons, 3, CV_32F);
	SCc = 1;

	CreateSmoothCons(weight);
}

asapWarp::~asapWarp(){}

// this calc the weights of 4 corners for each feature points
// eg. the weights of quad(i,j)
void asapWarp::SetControlPts(vector<Point> inputsPts, vector<Point> outputsPts)
{
	int len = inputsPts.size();
	dataterm_element_orgPt = inputsPts;
	dataterm_element_desPt = outputsPts;

	for (int i = 0; i < len; i++)
	{
		Point pt(inputsPts[i].x, inputsPts[i].y);
		dataterm_element_i.push_back(floor(pt.y/this.quadHeight)+1);
		dataterm_element_j.push_back(floor(pt.y/this.quadWidth)+1);

		Quad qd = source.getQuad(dataterm_element_i[i], dataterm_element_j[i]);

		float coefficients[4] = {};
		qd.getBilinearCoordinates(pt, coefficients);
		dataterm_element_V00.push_back(coefficients[0]);
		dataterm_element_V01.push_back(coefficients[1]);
		dataterm_element_V10.push_back(coefficients[2]);
		dataterm_element_V11.push_back(coefficients[3]);
	}
}

void asapWarp::Solve()
{
	Mat b = CreateDataCons();
	int N = SmoothConstraints.rows + DataConstraints.rows;

	Mat ARows = Mat::zeros(N, 1, CV_32F);
	Mat ACols = Mat::zeros(N, 1, CV_32F);
	Mat AVals = Mat::zeros(N, 1, CV_32F);

	int cc = 0;
	for (int i = 0; i < SmoothConstraints.rows; i++)
	{
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		cc++;
	}
	for (int i = 0; i < DataConstraints.rows; i++)
	{
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		ARows.at<CV_32F>(i, 1) = SmoothConstraints(i, 1)+1;
		cc++;
	}

	// TODO: solve the linear system w/ CUDA

	int halfcolumns = this.columns/2;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			Point pt(x.at<CV_32F>(i*width+j+1, 1), x.at<CV_32F>(halfcolumns+i*width+j+1, 1));
			destin.setVertex(i,j,pt);
		}
}

Mat asapWarp::Warp(Mat Img, int gap)
{
	warpIm = Mat::zeros(imgHeight+gap*2, imgWidth+gap*2, CV_32FC3);

	for(int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			Point p0 = source.getVertex(i-1, j-1);
			Point p1 = source.getVertex(i-1, j);
			Point p2 = source.getVertex(i, j-1);
			Point p3 = source.getVertex(i, j);

			Point p0 = source.getVertex(i-1, j-1);
			Point q1 = source.getVertex(i-1, j);
			Point q2 = source.getVertex(i, j-1);
			Point q3 = source.getVertex(i, j);

			Quad qd1(p0, p1, p2, p3);
			Quad qd2(q0, q1, q2, q3);
			quadWarp(Img, qd1, qd2);
}

void asapWarp::CalcHomos(Mat** homos)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			Quad q1 = source.getQuad(i,j);
			Quad q2 = destin.getQuad(i,j);

			Mat src = Mat::zeros()
		}
}

void getSmoothWeight(cv::Point uv, cv::Point V1, cv::Point V2, cv::Point V3);
Mat asapWarp::CreateSmoothCons(float weight);
Mat asapWarp::CreateDataCons();
// the triangles
void addCoefficient1(int i, int j, float weight);
void addCoefficient2(int i, int j, float weight);
void addCoefficient3(int i, int j, float weight);
void addCoefficient4(int i, int j, float weight);
void addCoefficient5(int i, int j, float weight);
void addCoefficient6(int i, int j, float weight);
void addCoefficient7(int i, int j, float weight);
void addCoefficient8(int i, int j, float weight);
void quadWarp(Mat im, Quad q1, Quad q2);