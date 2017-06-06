#pragma once

#include <vector>

#include "opencv2/core.hpp"

//#include "mesh.h"
//#include "quad.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

class asapWarp
{
public:
	asapWarp(int height, int width, int quadWidth, int quadHeight, int cellheight, int cellwidth, double weight);
	~asapWarp();
	void SetControlPts(vector<Point> inputsPts, vector<Point> outputsPts);
	void Solve();
	Mat Warp(Mat Img, int gap);
	void CalcHomos(Mat **homos);
    void PrintConstraints();
    void PrintVertex();

    Mat Constraints;

	// smooth constraints
	Mat SmoothConstraints;
	float SCc;
	int num_smooth_cons;

	// data constraints
	// TODO: convert vector into ptr
    vector<int> dataterm_element_i;
    vector<int> dataterm_element_j;
    vector<Point> dataterm_element_orgPt;
    vector<Point> dataterm_element_desPt;
    vector<float> dataterm_element_V00;
    vector<float> dataterm_element_V01;
    vector<float> dataterm_element_V10;
    vector<float> dataterm_element_V11;
    Mat DataConstraints;
    float DCc;
    int num_data_cons;
        
    int rowCount;
    int columns;
    
    vector<int> x_index;
    vector<int> y_index;
    
    // mesh
 //   Mesh source, destin;
    vector<Point2d> cellPts;
    
    // control points
    vector<Point> sourcePts, targetPts;
    
    int height,width; // mesh height,mesh width
    int quadWidth,quadHeight; // quadWidth,%quadHeight
    int imgHeight,imgWidth; // imgHeight,imgWidth
    
    Mat warpIm;
    float gap;

private:
	int CreateSmoothCons(float weight);
	int CreateDataCons();
	// the triangles
	void addCoefficient(int & cons, int i1, int j1, int i2, int j2, int i3, int j3, double weight);
	// void quadWarp(cv::Mat im, Quad q1, Quad q2);
	// compute position by index
	Point2d compute_pos(int i, int j);
	Point2d compute_uv(const Point2d V1, const Point2d V2, const Point2d V3);
	int index_x(int i, int j);
	int index_y(int i, int j);
};