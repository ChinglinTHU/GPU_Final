#pragma once

#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/calib3d.hpp"
//#include "opencv2/nonfree.hpp"

//#include "mesh.h"
//#include "quad.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;
typedef vector<vector<Mat> > BundleHomo;

class asapWarp
{
public:
    asapWarp();
	asapWarp(int height, int width, int cellheight, int cellwidth, float weight);
	~asapWarp();
	void SetControlPts(vector<Point2f> prevPts, vector<Point2f> nowPts, Mat globalH);
	void Solve();
    void IterativeSolve(int iter = 20);
    void SolvePoints(vector<vector<Point2f>> &prePts, vector<vector<Point2f>> &curPts);
	void CalcHomos(BundleHomo & homos);
    void PrintConstraintsSparse();
    void PrintConstraints(bool all);
    void PrintVertex();

    Point2f compute_pos(int i, int j);
    
    Mat Constraints;
    Mat Constants;

    int *SmoothRow, *SmoothCol;
    float* SmoothVal;
    int *ARow, *ACol;
    float* AVal;
    float* B;

	// smooth constraints
	int num_smooth_cons;
	// data constraints
    int num_data_cons;
        
    int rowCount;
    int columns;
    
    vector<Point2f> now_point, prev_point;
    float *weightsum;
    vector<Point2f> prev_feature, now_feature;
    
    vector<Point2f> cellPts;
    int allVertexNum;

    // control points
    // vector<Point> sourcePts, targetPts;
    
    float weight;
    int height,width; // mesh height,mesh width
    int quadWidth,quadHeight; // quadWidth,%quadHeight
    int imgHeight,imgWidth; // imgHeight,imgWidth
    Mat globalH;
    
    //Mat warpIm;
    //float gap;

private:
	int CreateSmoothCons(float weight);
    int CreateSmoothConsSparse(float weight);
	// the triangles
	void addSmoothCoefficient(int & cons, int i1, int j1, int i2, int j2, int i3, int j3, float weight);
    void addSmoothCoefficientSparse(int & cons, int & I, int i1, int j1, int i2, int j2, int i3, int j3, float weight);
    void addDataCoefficient(int & cons, Point2f pts, Point2f pts2);
    void addDataCoefficientSparse(int & cons, int & I, Point2f prev_pt, Point2f now_pt);

    float solveDataTerm(Point2f prev_pt, Point2f now_pt, float gamma);
    float solveSmoothTerm(int i1, int j1, int i2, int j2, int i3, int j3, float weight, float gamma);

	// void quadWarp(cv::Mat im, Quad q1, Quad q2);
	// compute position by index
	
	Point2f compute_uv(const Point2f V1, const Point2f V2, const Point2f V3);
	int index_x(int i, int j);
	int index_y(int i, int j);
    int index(int i, int j);
};