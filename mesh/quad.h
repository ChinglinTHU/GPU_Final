#pragma once

#include <vector>

#include "opencv2/core.hpp"

class Quad
{
public:
	Quad(cv::Point v00, cv::Point v01, cv::Point v10, cv::Point v11);
	~Quad();
	bool isPointIn(cv::Point pt);
	bool isPointsIn(std::vector<cv::Point> pts);
	bool getBilinearCoordinates(cv::Point pt, float* coefficients);
	float getMinX();
	float getMaxX();
	float getMinY();
	float getMaxY();
	bool isPointInTriangular(cv::Point pt, cv::Point V0, cv::Point V1, cv::Point V2);
	// bool isPointsInTriangular(std::vector<cv::Point> pts, cv::Point v0, cv::Point v1, cv::Point v2);

	cv::Point V00, V01, V10, V11;
};