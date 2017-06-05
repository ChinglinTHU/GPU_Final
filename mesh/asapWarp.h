#pragma once

#include <vector>

#include "opencv2/core.hpp"

#include "mesh.h"
#include "quad.h"

class asapWarp
{
public:
	asapWarp(int height, int width, int quadWidth, int quadHeight, float weight);
	~asapWarp();
	void SetControlPts(std::vector<cv::Point> inputsPts, std::vector<cv::Point> outputsPts);
	void Solve();
	cv::Mat Warp(cv::Mat Img, int gap);
	void CalcHomos(cv::Mat **homos);

	// smooth constraints
	cv::Mat SmoothConstraints;
	float SCc;
	int num_smooth_cons;

	// data constraints
	// TODO: convert vector into ptr
    std::vector<int> dataterm_element_i;
    std::vector<int> dataterm_element_j;
    std::vector<cv::Point> dataterm_element_orgPt;
    std::vector<cv::Point> dataterm_element_desPt;
    std::vector<float> dataterm_element_V00;
    std::vector<float> dataterm_element_V01;
    std::vector<float> dataterm_element_V10;
    std::vector<float> dataterm_element_V11;
    cv::Mat DataConstraints;
    float DCc;
    int num_data_cons;
        
        
    int rowCount;
    int columns;
    
    std::vector<int> x_index;
    std::vector<int> y_index;
    
    // mesh
    Mesh source, destin;
    
    // control points
    std::vector<cv::Point> sourcePts, targetPts;
    
    
    int height,width; // mesh height,mesh width
    int quadWidth,quadHeight; // quadWidth,%quadHeight
    int 0imgHeight,imgWidth; // imgHeight,imgWidth
    
    cv::Mat warpIm;
    float gap;

private:
	void getSmoothWeight(cv::Point uv, cv::Point V1, cv::Point V2, cv::Point V3);
	cv::Mat CreateSmoothCons(float weight);
	cv::Mat CreateDataCons();
	// the triangles
	void addCoefficient1(int i, int j, float weight);
	void addCoefficient2(int i, int j, float weight);
	void addCoefficient3(int i, int j, float weight);
	void addCoefficient4(int i, int j, float weight);
	void addCoefficient5(int i, int j, float weight);
	void addCoefficient6(int i, int j, float weight);
	void addCoefficient7(int i, int j, float weight);
	void addCoefficient8(int i, int j, float weight);
	void quadWarp(cv::Mat im, Quad q1, Quad q2);
};