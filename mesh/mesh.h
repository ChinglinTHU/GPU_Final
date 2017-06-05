#pragma once

#include "opencv2/core.hpp"
#include "quad.h"

class Mesh
{
public:
	Mesh(int rows, int cols, int quadWidth, int quadHeight);
	~Mesh();
	cv::Point getVertex(int i, int j);
	void setVertex(int i, int j, cv::Point pt);
	Quad getQuad(int i, int j);

	int imgRows, imgCols;
	int meshWidth, meshHeight;
	int quadWidth, quadHeight;
	cv::Mat xMat, yMat;
};