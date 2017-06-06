#include "asapWarp.h"

#include <cmath>

using namespace std;
using namespace cv;
using namespace cv::cuda;

typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;

asapWarp::asapWarp(int height, int width, int cellheight, int cellwidth, double weight)
{

	this->imgHeight = height;
	this->imgWidth = width;
	this->quadWidth = width / (cellwidth - 1);
	this->quadHeight = height / (cellheight - 1);
	this->height = cellheight;
	this->width = cellwidth;

	// each cell only got 4 triangles with x, y -> total 8 equations one cell
	num_smooth_cons = (this->height-1)*(this->width-1)*8;
	allVertexNum = this->height*this->width;
	columns = allVertexNum*2;
	num_data_cons = 0;
	this->Constraints = Mat::zeros(num_smooth_cons, columns, CV_64FC1);

	CreateSmoothCons(weight);
}

asapWarp::~asapWarp(){}

// this calc the weights of 4 corners for each feature points
// eg. the weights of quad(i,j)
void asapWarp::SetControlPts(vector<KeyPoint> prevPts, vector<KeyPoint> nowPts, vector<DMatch> match)
{
	int len = match.size();
	num_data_cons = len*2;

	// Copy smooth matrix to all
	Mat allConstraints = Mat::zeros(num_smooth_cons + num_data_cons, columns, CV_64FC1);
	Constraints.copyTo(allConstraints(Rect(0, 0, Constraints.cols, Constraints.rows)));
	Constraints = allConstraints;

	Constants = Mat::zeros(num_smooth_cons + num_data_cons, 1, CV_64FC1);

	int ind_x, ind_y;
	int cons = num_smooth_cons;
	for (int i = 0; i < len; i++)
	{
		Point2d prevPt = prevPts[match[i].queryIdx].pt;
		Point2d nowPt = nowPts[match[i].trainIdx].pt;

		addDataCoefficient(cons, prevPt, nowPt);
	}
	
}

void asapWarp::Solve()
{
	// TODO: solve the linear system w/ CUDA
	Mat x;
	bool valid = solve(Constraints, Constants, x, DECOMP_SVD);
	
	cellPts = vector<Point2d>(allVertexNum);
	for (int i = 0; i < allVertexNum; ++i)
		cellPts[i] = Point2d(x.at<double>(2*i, 0)+compute_pos(i%width, i/width).x, x.at<double>(2*i+1, 0)+compute_pos(i%width, i/width).y);

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

void asapWarp::CalcHomos(Mat homos)
{
//	homos = Mat::zeros(height-1, width-1, CV_64FC(9));
	vector<Point2d> V(4);
	vector<Point2d> W(4);
	Mat h;
	for (int i = 0; i < height-1; i++)
		for (int j = 0; j < width-1; j++)
		{
			V[0] = compute_pos(i, j);
			V[1] = compute_pos(i, j+1);
			V[2] = compute_pos(i+1, j);
			V[3] = compute_pos(i+1, j+1);
			W[0] = cellPts[j*width + i];
			W[1] = cellPts[(j+1)*width + i];
			W[2] = cellPts[j*width + (i+1)];
			W[3] = cellPts[(j+1)*width + (i+1)];

			h = findHomography(V, W);
			homos.at<double>(i, j) = 0;

			//printf("%d, %d, %d\n", homos.cols, homos.rows, homos.channels());
			//printf("%lf ", homos.at<Vec9d>(i, j)[5]);

			homos.at<Vec9d>(i, j)[0] = h.at<double>(0, 0);
			homos.at<Vec9d>(i, j)[1] = h.at<double>(0, 1);
			homos.at<Vec9d>(i, j)[2] = h.at<double>(0, 2);
			homos.at<Vec9d>(i, j)[3] = h.at<double>(1, 0);
			homos.at<Vec9d>(i, j)[4] = h.at<double>(1, 1);
			homos.at<Vec9d>(i, j)[5] = h.at<double>(1, 2);
			homos.at<Vec9d>(i, j)[6] = h.at<double>(2, 0);
			homos.at<Vec9d>(i, j)[7] = h.at<double>(2, 1);
			homos.at<Vec9d>(i, j)[8] = h.at<double>(2, 2);
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
	printf("\n");	
}

void asapWarp::PrintConstraints(bool all)
{
	printf("smoothConstraints: %d\n", num_smooth_cons);
	double t;
	for (int i = 0; i < num_smooth_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<double>(i, j);
			if (!all)
			{
				if (t != 0)
					printf("(%d, %d) %.2lf\n", i, j, t);
			}
			else
			{
				if(j == 0)
					printf("%.2lf | ", Constants.at<double>(i, 0));
				printf("%.2lf ", t);
				if(j == columns-1)
					printf("\n");
			}
		}
	printf("dataConstraints: %d\n", num_data_cons);
	for (int i = num_smooth_cons; i < num_smooth_cons+num_data_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<double>(i, j);
			if (!all)
			{
				if (t != 0)
					printf("(%d, %d) %.2lf\n", i, j, t);
			}
			else
			{
				if(j == 0)
					printf("%.2lf | ", Constants.at<double>(i, 0));
				printf("%.2lf ", t);
				if(j == columns-1)
					printf("\n");
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
			addSmoothCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j+1;
			i2 = i+1; 	j2 = j;
			i3 = i; 	j3 = j;
			addSmoothCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i; 	j1 = j+1;
			i2 = i+1; 	j2 = j+1;
			i3 = i+1; 	j3 = j;
			addSmoothCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j;
			i2 = i; 	j2 = j;
			i3 = i; 	j3 = j+1;
			addSmoothCoefficient(cons, i1, j1, i2, j2, i3, j3, weight);
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

// the triangles
void asapWarp::addSmoothCoefficient(int & cons, int i1, int j1, int i2, int j2, int i3, int j3, double weight)
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

void asapWarp::addDataCoefficient(int & cons, Point2d prev_pt, Point2d now_pt)
{
	double x = prev_pt.x;
	double y = prev_pt.y;
	int ind_x = int(x) / quadWidth;
	int ind_y = int(y) / quadHeight;
	
	if (ind_x < width-1 && ind_y < height-1)
	{
		Point2d V00 = compute_pos(ind_x, ind_y);
		Point2d V11 = compute_pos(ind_x+1, ind_y+1);

		double u = (x - V00.x) / (V11.x - V00.x);
		double v = (y - V00.y) / (V11.y - V00.y);
	
		cons++;
		Constraints.at<double>(cons, index_x(ind_x, ind_y))     = (1-u)*(1-v);
		Constraints.at<double>(cons, index_x(ind_x, ind_y+1))   = (1-u)*v;
		Constraints.at<double>(cons, index_x(ind_x+1, ind_y))   = u*(1-v);
		Constraints.at<double>(cons, index_x(ind_x+1, ind_y+1)) = u*v;
		Constants.at<double>(cons, 0) 							= now_pt.x - prev_pt.x;
		cons++;
		Constraints.at<double>(cons, index_y(ind_x, ind_y))     = (1-u)*(1-v);
		Constraints.at<double>(cons, index_y(ind_x, ind_y+1))   = (1-u)*v;
		Constraints.at<double>(cons, index_y(ind_x+1, ind_y))   = u*(1-v);
		Constraints.at<double>(cons, index_y(ind_x+1, ind_y+1)) = u*v;
		Constants.at<double>(cons, 0) 							= now_pt.y - prev_pt.y;
	}
}
// void quadWarp(Mat im, Quad q1, Quad q2);