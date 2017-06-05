// the meshed image
#include "mesh.h"

#include <vector>

using namespace std;
using namespace cv;

Mesh::Mesh(int rows, int cols, int quadWidth, int quadHeight)
{
	imgRows = rows;
	imgCols = cols;
	this.quadWidth = quadWidth;
	this.quadHeight = quadHeight;

	vector<int> xSet, ySet;

	int x = 0;
	while(imgCols - x > 0.5 * quadWidth)
	{
		xSet.push_back(x);
		x += quadWidth;
	}
	xSet.push_back(x);

	int y = 0;
	while(imgRows - y > 0.5 * quadHeight)
	{
		ySet.push_back(y);
		y += quadHeight;
	}
	ySet.push_back(y);

	meshWidth = xSet.size();
	meshHeight = ySet.size();

	// the coord of the mesh on the img
	int type = CV_16U;
	xMat = Mat::zeros(meshHeight, meshWidth, type);
	yMat = Mat::zeros(meshHeight, meshWidth, type);

	for(int y = 0; y < meshHeight; y++)
		for(int x = 0; x < meshWidth; x++)
		{
			xMat.at<type>(y, x) = xSet[x];
			yMat.at<type>(y, x) = ySet[y];
		}
}

Mesh::~Mesh(){}

Point Mesh::getVertex(int i, int y)
{
	int type = CV_16U;
	return Point(xMat.at<type>(i+1, j+1), yMat.at<type>(i+1, j+1));
}

void Mesh::setVertex(int i, int j, Point pt)
{
	int type = CV_16U;
	xMat.at<type>(i+1, j+1) = pt.x;
	yMat.at<type>(i+1, j+1) = pt.y;
}

Quad Mesh::getQuad(int i, int j)
{
	Point V00 = getVertex(i-1, j-1);
	Point V01 = getVertex(i-1, j);
	Point V10 = getVertex(i, j-1);
	Point V11 = getVertex(i, j);
	return Quad(V00, V01, V10, V11);
}