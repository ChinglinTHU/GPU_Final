#include "asapWarp.h"

#include <cmath>

asapWarp::asapWarp(int height, int width, int quadWidth, int quadHeight, int cellheight, int cellwidth, double weight)
{

	this->imgHeight = height;
	this->imgWidth = width;
	this->quadWidth = quadWidth;
	this->quadHeight = quadHeight;
	this->height = cellheight;
	this->width = cellwidth;
	printf("cellh = %d, cellw = %d\n", cellheight, cellwidth);

	// each cell only got 4 triangles with x, y -> total 8 equations one cell
	num_smooth_cons = (this->height-1)*(this->width-1)*8;
	columns = this->height*this->width*2;
	num_data_cons = 0;
	this->Constraints = Mat::zeros(num_smooth_cons + num_data_cons, columns, CV_64FC1);
	SCc = 1;

	CreateSmoothCons(weight);
}

asapWarp::~asapWarp(){}

// this calc the weights of 4 corners for each feature points
// eg. the weights of quad(i,j)
void asapWarp::SetControlPts(vector<Point> inputsPts, vector<Point> outputsPts)
{
	/*
	int len = inputsPts.size();
	dataterm_element_orgPt = inputsPts;
	dataterm_element_desPt = outputsPts;

	for (int i = 0; i < len; i++)
	{
		Point pt(inputsPts[i].x, inputsPts[i].y);
		dataterm_element_i.push_back(floor(pt.y/quadHeight)+1);
		dataterm_element_j.push_back(floor(pt.y/quadWidth)+1);

		Quad qd = source.getQuad(dataterm_element_i[i], dataterm_element_j[i]);

		float coefficients[4] = {};
		qd.getBilinearCoordinates(pt, coefficients);
		dataterm_element_V00.push_back(coefficients[0]);
		dataterm_element_V01.push_back(coefficients[1]);
		dataterm_element_V10.push_back(coefficients[2]);
		dataterm_element_V11.push_back(coefficients[3]);
	}
	*/
}

void asapWarp::Solve()
{
	//Mat b = CreateDataCons();
	//int N = SmoothConstraints.rows + DataConstraints.rows;

	//Mat ARows = Mat::zeros(N, 1, CV_32F);
	//Mat ACols = Mat::zeros(N, 1, CV_32F);
	//Mat AVals = Mat::zeros(N, 1, CV_32F);

	/*
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
	*/
	Mat b = Mat::zeros(columns, 1, CV_64FC1);
	Mat x;
	bool valid = solve(Constraints, b, x, DECOMP_SVD);
	
	int halfcolumns = this->columns/2;
	cellPts = vector<Point2d>(halfcolumns);
	for (int i = 0; i < halfcolumns; ++i)
	{
		cellPts[i] = Point2d(x.at<double>(2*i, 1), x.at<double>(2*i+1, 1));
	}
}

Mat asapWarp::Warp(Mat Img, int gap)
{
	/*
	warpIm = Mat::zeros(imgHeight+gap*2, imgWidth+gap*2, CV_32FC3);

	for(int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
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

	*/
}

void asapWarp::CalcHomos(Mat** homos)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			//Quad q1 = source.getQuad(i,j);
			//Quad q2 = destin.getQuad(i,j);

			// Mat src = Mat::zeros();
		}
}

void asapWarp::PrintVertex()
{
	for (int i = 0; i < cellPts.size(); ++i)
	{
		if(i % width == 0)
			printf("\n");
		printf("(%.2f, %.2f), ", cellPts[i].x, cellPts[i].y);
	}
	
}

void asapWarp::PrintConstraints()
{
	printf("smoothConstraints: %d\n", num_smooth_cons);
	double t;
	for (int i = 0; i < num_smooth_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<double>(i, j);
			if (t != 0)
			{
				printf("(%d, %d) %1.2lf\n", i, j, t);
			}
		}
	printf("dataConstraints: %d\n", num_data_cons);
	for (int i = num_smooth_cons; i < num_smooth_cons+num_data_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<double>(i, j);
			if (t != 0)
			{
				printf("(%d, %d) %1.2lf\n", i, j, t);
			}
		}
}

int asapWarp::CreateSmoothCons(float weight)
{
	// Constraints = Mat(num_smooth_cons + num_data_cons, this.height*this.width, CV_32F);
	int cons = -1;
	int i1, i2, i3, j1, j2, j3;
	Point2d V1, V2, V3, uv;
	float u, v;
	for (int i = 0; i < height-1; i++)
		for (int j = 0; j < width-1; j++)
		{
			i1 = i; 	j1 = j;
			i2 = i; 	j2 = j+1;
			i3 = i+1; 	j3 = j+1;
			addCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j+1;
			i2 = i+1; 	j2 = j;
			i3 = i; 	j3 = j;
			addCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i; 	j1 = j+1;
			i2 = i+1; 	j2 = j+1;
			i3 = i+1; 	j3 = j;
			addCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j;
			i2 = i; 	j2 = j;
			i3 = i; 	j3 = j+1;
			addCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
		}
	return cons+1;
}

int asapWarp::index_x(int i, int j)
{
	return 2*(j*width + i);
}

int asapWarp::index_y(int i, int j)
{
	return 2*(j*width + i)+1;
}

Point2d asapWarp::compute_pos(int i, int j)
{
	double x = i < this->width - 1 ? i * this->quadWidth : this->imgWidth;
	double y = j < this->height - 1 ? j * this->quadHeight : this->imgHeight;
	return Point2d(x, y);
}

Point2d asapWarp::compute_uv(const Point2d V1, const Point2d V2, const Point2d V3)
{
	Point2d V21 = V1 - V2;
	Point2d V23 = V3 - V2;
	double d1 = norm(V21);
	double d3 = norm(V23);
	double u = (V21.x*V23.x + V21.y*V23.y) / (d1 * d3);
	double v = sqrt(1-u*u);
	u = u * d1 / d3;
	v = v * d1 / d3;
	return Point2d(u, v);
}

int asapWarp::CreateDataCons()
{
	return 0;
}
// the triangles
void asapWarp::addCoefficient(int & cons, int i1, int j1, int i2, int j2, int i3, int j3, double weight)
{
	Point2d V1, V2, V3, uv;
	double u, v;
	V1 = compute_pos(i1, j1);
	V2 = compute_pos(i2, j2);
	V3 = compute_pos(i3, j3);
	uv = compute_uv(V1, V2, V3);
	u = uv.x;
	v = uv.y;
	cons++;
	Constraints.at<double>(cons, index_x(i2, j2)) = (1-u)*weight;
	Constraints.at<double>(cons, index_x(i3, j3)) = (u)*weight;
	Constraints.at<double>(cons, index_y(i3, j3)) = (v)*weight;
	Constraints.at<double>(cons, index_y(i2, j2)) = (-v)*weight;
	Constraints.at<double>(cons, index_x(i1, j1)) = -weight;
	cons++;
	Constraints.at<double>(cons, index_y(i2, j2)) = (1-u)*weight;
	Constraints.at<double>(cons, index_y(i3, j3)) = (u)*weight;
	Constraints.at<double>(cons, index_x(i2, j2)) = (v)*weight;
	Constraints.at<double>(cons, index_x(i3, j3)) = (-v)*weight;
	Constraints.at<double>(cons, index_y(i1, j1)) = -weight;
}
// void quadWarp(Mat im, Quad q1, Quad q2);