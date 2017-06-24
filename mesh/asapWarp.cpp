#include "asapWarp.h"

#include <iostream>
#include <cmath>

//#include "../utils/cuSpSolver.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

typedef Vec<float, 9> Vec9f;
typedef Vec<double, 9> Vec9d;
typedef vector<vector<Mat> > BundleHomo;

asapWarp::asapWarp(){}

asapWarp::asapWarp(int height, int width, int cellheight, int cellwidth, float weight)
{

	this->imgHeight = height;
	this->imgWidth = width;
	this->quadWidth = width / (cellwidth - 1);
	this->quadHeight = height / (cellheight - 1);
	this->height = cellheight;
	this->width = cellwidth;

	this->weight = weight;

	// each cell only got 4 triangles with x, y -> total 8 equations one cell
	num_smooth_cons = (this->height-1)*(this->width-1)*8;
	allVertexNum = this->height*this->width;
	columns = allVertexNum*2;
	num_data_cons = 0;
	this->Constraints = Mat::zeros(num_smooth_cons, columns, CV_32FC1);

	CreateSmoothCons(weight);

	// jacobi method
	vector<Point2f> now_point(allVertexNum);
	for (int i = 0; i < this->width; i++)
		for (int j = 0; j < this->height; j++)
			now_point[index(i, j)] = Point2f(0.f, 0.f);

	this->now_point = now_point;
	weightsum = new float[allVertexNum];
	for (int i = 0; i < allVertexNum; i++)
		weightsum[i] = 1;
}

asapWarp::~asapWarp(){}

// this calc the weights of 4 corners for each feature points
// eg. the weights of quad(i,j)
void asapWarp::SetControlPts(vector<Point2f> prevPts, vector<Point2f> nowPts, Mat h)
{
	int len = prevPts.size();
	num_data_cons = len*2;

	// Copy smooth matrix to all
	Mat allConstraints = Mat::zeros(num_smooth_cons + num_data_cons, columns, CV_32FC1);
	Constraints.copyTo(allConstraints(Rect(0, 0, Constraints.cols, Constraints.rows)));
	Constraints = allConstraints;
	
	Constants = Mat::zeros(num_smooth_cons + num_data_cons, 1, CV_32FC1);
	
	///* with global
	globalH = h.clone();
	int ind_x, ind_y;
	int cons = num_smooth_cons-1;
	int sparse_cons = num_smooth_cons-1;
	int I = num_smooth_cons*4-1;
	vector<Point2f> warpNowPts;
	perspectiveTransform(nowPts, warpNowPts, globalH.inv());
	for (int i = 0; i < len; i++)
	{
		addDataCoefficient(cons, prevPts[i], warpNowPts[i]);
	}

	prev_feature = prevPts;
	now_feature = warpNowPts;
	//*/
}

void asapWarp::Solve()
{
	Mat x;

	bool valid = solve(Constraints, Constants, x, DECOMP_QR);

	Mat diff = Constraints*x-Constants;
	float loss = 0.f;
	
 	for (int i = 0; i < diff.rows; i++)
		loss += diff.at<float>(i, 0)*diff.at<float>(i, 0);
	//cout << "QR solve loss = " << loss << endl;

	for (int i = 0; i < allVertexNum; ++i)
	{
		cellPts.push_back(Point2f(x.at<float>(2*i, 0)+compute_pos(i%width, i/width).x, x.at<float>(2*i+1, 0)+compute_pos(i%width, i/width).y));
	}

	/*
	cout << "solve()" << endl;
	for (int i = 0; i < cellPts.size(); i++)
		cout << cellPts[i] << endl;
	//*/

	perspectiveTransform(cellPts, cellPts, globalH);
}

void asapWarp::IterativeSolve(int iter)
{
	float gamma = .01;
	float loss, prev_loss = 1.f;
	for (int step = 0; step < iter; step++)
	{
		//gamma *= 0.9;
		/*
		cout << "============== step: " << step << endl;
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
				cout << "(" << i << ", " << j << "): " << now_point[index(i, j)]+compute_pos(i, j) << endl;
		//*/

		loss = 0;

		prev_point = now_point;
		int i1, i2, i3, j1, j2, j3;
		Point2f V1, V2, V3, uv;
		float u, v;
		for (int i = 0; i < width-1; i++)
			for (int j = 0; j < height-1; j++)
			{
				i1 = i; 	j1 = j;
				i2 = i; 	j2 = j+1;
				i3 = i+1; 	j3 = j+1;
				loss += solveSmoothTerm(i1, j1, i2, j2, i3, j3, weight, gamma);
				i1 = i+1; 	j1 = j+1;
				i2 = i+1; 	j2 = j;
				i3 = i; 	j3 = j;
				loss += solveSmoothTerm(i1, j1, i2, j2, i3, j3, weight, gamma);
				i1 = i; 	j1 = j+1;
				i2 = i+1; 	j2 = j+1;
				i3 = i+1; 	j3 = j;
				loss += solveSmoothTerm(i1, j1, i2, j2, i3, j3, weight, gamma);
				i1 = i+1; 	j1 = j;
				i2 = i; 	j2 = j;
				i3 = i; 	j3 = j+1;
				loss += solveSmoothTerm(i1, j1, i2, j2, i3, j3, weight, gamma);
			}

		for (int i = 0; i < prev_feature.size(); i++)
		{
			loss += solveDataTerm(prev_feature[i], now_feature[i], gamma);
		}

		if (loss > prev_loss)
			break;

		/*
		for (int i = 0; i < now_point.size(); i++)
		{
			now_point[i] /= weightsum[i];
			weightsum[i] = 1;
		}
		*/
		prev_loss = loss;
	}


	for (int i = 0; i < now_point.size(); i++)
		now_point[i] += compute_pos(i%width, i/width);

	/*
	cout << "=====iterative====" << endl;
	for (int i = 0; i < now_point.size(); i++)
		cout << now_point[i] << endl;
	//*/

	perspectiveTransform(now_point, cellPts, globalH);
}

float asapWarp::solveDataTerm(Point2f prev_pt, Point2f now_pt, float gamma)
{
	float x = prev_pt.x;
	float y = prev_pt.y;
	int ind_x = int(x) / quadWidth;
	int ind_y = int(y) / quadHeight;

	int i = 2, j = 4;
	
	/*
	if (ind_x >= i-1 && ind_x <= i && ind_y >= j-1 && ind_y <= j)
	{
		cout << endl << "origin now_point = " << now_point[index(i, j)] / weightsum[index(i, j)] << endl;
		cout << prev_point[index(ind_x, ind_y)] << ", " << prev_point[index(ind_x+1, ind_y)] << endl;
		cout << prev_point[index(ind_x, ind_y+1)] << ", " << prev_point[index(ind_x+1, ind_y+1)] << endl;
	}
	//*/

	if (ind_x < width-1 && ind_y < height-1)
	{
		Point2f V00 = compute_pos(ind_x, ind_y);
		Point2f V11 = compute_pos(ind_x+1, ind_y+1);

		float u = (x - V00.x) / (V11.x - V00.x);
		float v = (y - V00.y) / (V11.y - V00.y);

		/*
		if (u > .1f && u < .9f && v > .1f && v < .9f)
		{
			now_point[index(ind_x, ind_y)] += 1/((1-u)*(1-v))*(now_pt - prev_point[index(ind_x+1, ind_y+1)]*(u)*(v) 
											- prev_point[index(ind_x, ind_y+1)]*(1-u)*v - prev_point[index(ind_x+1, ind_y)]*(u)*(1-v));
			weightsum[index(ind_x, ind_y)] += 1;

			now_point[index(ind_x+1, ind_y)] += 1/((u)*(1-v))*(now_pt - prev_point[index(ind_x+1, ind_y+1)]*(u)*(v) 
											- prev_point[index(ind_x, ind_y+1)]*(1-u)*v - prev_point[index(ind_x, ind_y)]*(1-u)*(1-v));
			weightsum[index(ind_x+1, ind_y)] += 1;

			now_point[index(ind_x, ind_y+1)] += 1/((1-u)*v)*(now_pt - prev_point[index(ind_x+1, ind_y+1)]*(u)*(v) 
											- prev_point[index(ind_x, ind_y)]*(1-u)*(1-v) - prev_point[index(ind_x+1, ind_y)]*(u)*(1-v));
			weightsum[index(ind_x, ind_y+1)] += 1;

			now_point[index(ind_x+1, ind_y+1)] += 1/((u)*(v))*(now_pt - prev_point[index(ind_x, ind_y)]*(1-u)*(1-v)
											- prev_point[index(ind_x, ind_y+1)]*(1-u)*v - prev_point[index(ind_x+1, ind_y)]*(u)*(1-v));
			weightsum[index(ind_x+1, ind_y+1)] += 1;
		}
		*/
		float w = 0.1f;

		/*
		now_point[index(ind_x, ind_y)] += prev_point[index(ind_x, ind_y)] + (now_pt - prev_pt)*w;
		weightsum[index(ind_x, ind_y)] += 1;

		now_point[index(ind_x+1, ind_y)] += prev_point[index(ind_x+1, ind_y)] + (now_pt - prev_pt)*w;
		weightsum[index(ind_x+1, ind_y)] += 1;

		now_point[index(ind_x, ind_y+1)] += prev_point[index(ind_x, ind_y+1)] + (now_pt - prev_pt)*w;
		weightsum[index(ind_x, ind_y+1)] += 1;

		now_point[index(ind_x+1, ind_y+1)] += prev_point[index(ind_x+1, ind_y+1)] + (now_pt - prev_pt)*w;
		weightsum[index(ind_x+1, ind_y+1)] += 1;
		*/

		Point2f loss = prev_point[index(ind_x, ind_y)]*(1-u)*(1-v) + prev_point[index(ind_x+1, ind_y)]*(u)*(1-v)
					 + prev_point[index(ind_x, ind_y+1)]*(1-u)*(v) + prev_point[index(ind_x+1, ind_y+1)]*(u)*(v) 
					 - (now_pt - prev_pt);
		now_point[index(ind_x, ind_y)] -= gamma*loss*(1-u)*(1-v);
		now_point[index(ind_x+1, ind_y)] -= gamma*loss*(u)*(1-v);
		now_point[index(ind_x, ind_y+1)] -= gamma*loss*(1-u)*(v);
		now_point[index(ind_x+1, ind_y+1)] -= gamma*loss*(u)*(v);

		/*
		if (ind_x >= i-1 && ind_x <= i && ind_y >= j-1 && ind_y <= j)
		{
			cout << "u = " << u << ", v = " << v << endl;
			cout << "prev_pt = " << prev_pt << endl;
			cout << "now_pt = " << now_pt << endl;
			cout << "solve data term = " << now_point[index(i, j)] / weightsum[index(i, j)] << endl;

		}
		//*/
		return loss.x*loss.x + loss.y*loss.y;	
	}
	
	return 0.f;
}

float asapWarp::solveSmoothTerm(int i1, int j1, int i2, int j2, int i3, int j3, float weight, float gamma)
{
	Point2f V1, V2, V3, V3_2;
	float u, v;
	V1 = prev_point[index(i1, j1)] + compute_pos(i1, j1);
	V2 = prev_point[index(i2, j2)] + compute_pos(i2, j2);
	V3 = prev_point[index(i3, j3)] + compute_pos(i3, j3);
	V3_2 = V3-V2;

	///* simple smooth coefficients
	float s = norm(compute_pos(i1, j1) - compute_pos(i2, j2))/norm(compute_pos(i2, j2) - compute_pos(i3, j3));
	int i = 2, j = 4;

	/*
	if (i1 == i && j1 == j)
		cout << endl << "origin: now_point = " << now_point[index(i1, j1)] / weightsum[index(i1, j1)] << endl;
		//*/

	/*
	now_point[index(i1, j1)] += weight*V2 + s*weight*Point2f(V3_2.y, -V3_2.x);
								
	weightsum[index(i1, j1)] += weight;
	*/
	Point2f loss = (V1-V2) - s*Point2f(V3_2.y, -V3_2.x);

	now_point[index(i1, j1)] -= gamma*loss;
	now_point[index(i2, j2)] -= gamma*Point2f(loss.x*(-loss.x-s*loss.y), loss.y*(s*loss.x-loss.y));
	now_point[index(i3, j3)] -= gamma*Point2f(loss.x*(s*loss.y), loss.y*(-s*loss.x));

	/*
	if (i1 == i && j1 == j)
	{
		cout << "s = " << s << endl;
		cout << "V2 = " << V2 << endl;
		cout << "V3 = " << V3 << endl;
		cout << "s*Point2f(V3_2.y, V3_2.x) = " << s*Point2f(V3_2.y, V3_2.x) << endl;
		//cout << "solve smooth term = " << now_point[index(i1, j1)] / weightsum[index(i1, j1)] << endl;
		cout << "solve smooth term = " << now_point[index(i1, j1)] << endl;
	}
	//*/

	return loss.x*loss.x+loss.y*loss.y;
}

void asapWarp::SolvePoints(vector<vector<Point2f>> &prePts, vector<vector<Point2f>> &curPts)
{
	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
		{
			Point2f p = cellPts[j*width+i];
			// cout<<j<<" "<<i<<endl;
			p.x = p.x < 0 ? 0 : p.x;
			p.x = p.x >= imgWidth ? imgWidth-1 : p.x;
			p.y = p.y < 0 ? 0 : p.y;
			p.y = p.y >= imgHeight ? imgHeight-1 : p.y;
			// cout<<p<<endl;

			// calc which cell this cellPoint belong to
			int celli = (int)(p.x / quadWidth);
			int cellj = (int)(p.y / quadHeight);
			// cout<<celli<<", "<<cellj<<endl;
			// calc uv, the w is counted clockwise
			float u = cellPts[j*width+i].x / (float)(quadWidth) - (float)(celli);
			float v = cellPts[j*width+i].y / (float)(quadHeight) - (float)(cellj);
			// cout<<u<<","<<v<<endl;
			float w0 = (1-u)*(1-v);
			float w1 = u*(1-v);
			float w2 = (1-u)*v;
			float w3 = u*v;

			// calc the weighted cellPoint according to the prePts
			p =  prePts[celli][cellj]*w0;
			p += prePts[celli+1][cellj]*w1;
			p += prePts[celli][cellj+1]*w2;
			p += prePts[celli+1][cellj+1]*w3;

			curPts[i][j] = p;
			// cout<<p<<endl;
		}
}

void asapWarp::CalcHomos(BundleHomo & homos)
{
//	homos = Mat::zeros(height-1, width-1, CV_32FC(9));
	vector<Point2f> V(4);
	vector<Point2f> W(4);
	Mat h;

	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
		{
			V[0] = compute_pos(i, j);
			V[1] = compute_pos(i, j+1);
			V[2] = compute_pos(i+1, j);
			V[3] = compute_pos(i+1, j+1);
			W[0] = cellPts[j*width + i];
			W[1] = cellPts[(j+1)*width + i];
			W[2] = cellPts[j*width + (i+1)];
			W[3] = cellPts[(j+1)*width + (i+1)];

			// findHomography return mat with double type
			h = findHomography(V, W);

			h.convertTo(homos[i][j], CV_32FC1);
		}
}

void asapWarp::PrintVertex()
{
	cout << "print vertex: " << cellPts.size() << endl;
	for (int i = 0; i < cellPts.size(); ++i)
	{
		if(i % width == 0)
			printf("\n");
		printf("(%.2f, %.2f), ", cellPts[i].x, cellPts[i].y);
	}
	printf("\n");	
}

void asapWarp::PrintConstraintsSparse()
{
	printf("Print Sparse constraints\n");
	for (int i = 0; i < 4*(num_smooth_cons+num_data_cons); i++)
		printf("(%d, %d) %.2lf\n", ARow[i], ACol[i], AVal[i]);

	printf("Print Sparse Constant\n");
	for (int i = 0; i < num_smooth_cons+num_data_cons; i++)
		printf("B[%d] = %.2lf\n", i, B[i]);
}

void asapWarp::PrintConstraints(bool all)
{
	printf("Print solve() constraints\n");
	printf("smoothConstraints: %d\n", num_smooth_cons);
	float t;
	for (int i = 0; i < num_smooth_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<float>(i, j);
			if (!all)
			{
				if (t != 0)
					printf("(%d, %d) %.2lf\n", i, j, t);
			}
			else
			{
				if(j == 0)
					printf("%.2lf | ", Constants.at<float>(i, 0));
				printf("%.2lf ", t);
				if(j == columns-1)
					printf("\n");
			}
		}
	printf("dataConstraints: %d\n", num_data_cons);
	for (int i = num_smooth_cons; i < num_smooth_cons+num_data_cons; ++i)
		for (int j = 0; j < columns; ++j)
		{
			t = Constraints.at<float>(i, j);
			if (!all)
			{
				if (t != 0)
					printf("(%d, %d) %.2lf\n", i, j, t);
			}
			else
			{
				if(j == 0)
					printf("%.2lf | ", Constants.at<float>(i, 0));
				printf("%.2lf ", t);
				if(j == columns-1)
					printf("\n");
			}
		}

	printf("Constant\n");
	for (int i = 0; i < num_smooth_cons+num_data_cons; i++)
		printf("B[%d] = %.2lf\n", i, Constants.at<float>(i, 0));
}

int asapWarp::CreateSmoothConsSparse(float weight)
{
	// Constraints = Mat(num_smooth_cons + num_data_cons, this.height*this.width, CV_32F);
	int cons = -1;
	int I = -1;
	int i1, i2, i3, j1, j2, j3;
	Point2f V1, V2, V3, uv;
	float u, v;
	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
		{
			i1 = i; 	j1 = j;
			i2 = i; 	j2 = j+1;
			i3 = i+1; 	j3 = j+1;
			addSmoothCoefficientSparse(cons, I, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j+1;
			i2 = i+1; 	j2 = j;
			i3 = i; 	j3 = j;
			addSmoothCoefficientSparse(cons, I, i1, j1, i2, j2, i3, j3, weight);
			i1 = i; 	j1 = j+1;
			i2 = i+1; 	j2 = j+1;
			i3 = i+1; 	j3 = j;
			addSmoothCoefficientSparse(cons, I, i1, j1, i2, j2, i3, j3, weight);
			i1 = i+1; 	j1 = j;
			i2 = i; 	j2 = j;
			i3 = i; 	j3 = j+1;
			addSmoothCoefficientSparse(cons, I, i1, j1, i2, j2, i3, j3, weight);
		}
	return cons+1;
}

int asapWarp::CreateSmoothCons(float weight)
{
	// Constraints = Mat(num_smooth_cons + num_data_cons, this.height*this.width, CV_32F);
	int cons = -1;
	int i1, i2, i3, j1, j2, j3;
	Point2f V1, V2, V3, uv;
	float u, v;
	for (int i = 0; i < width-1; i++)
		for (int j = 0; j < height-1; j++)
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

int asapWarp::index(int i, int j)
{
	return (j*width + i);
}

int asapWarp::index_x(int i, int j)
{
	return 2*(j*width + i);
}

int asapWarp::index_y(int i, int j)
{
	return 2*(j*width + i)+1;
}

Point2f asapWarp::compute_pos(int i, int j)
{
	float x = i < this->width - 1 ? i * this->quadWidth : this->imgWidth;
	float y = j < this->height - 1 ? j * this->quadHeight : this->imgHeight;
	return Point2f(x, y);
}

Point2f asapWarp::compute_uv(const Point2f V1, const Point2f V2, const Point2f V3)
{
	Point2f V21 = V1 - V2;
	Point2f V23 = V3 - V2;
	float d1 = norm(V21);
	float d3 = norm(V23);
	float u = (V21.x*V23.x + V21.y*V23.y) / (d1 * d3);
	float v = sqrt(1-u*u);
	u = u * d1 / d3;
	v = v * d1 / d3;
	return Point2f(u, v);
}

void asapWarp::addSmoothCoefficientSparse(int & cons, int & I, int i1, int j1, int i2, int j2, int i3, int j3, float weight)
{
	Point2f V1, V2, V3, uv;
	float u, v;
	V1 = compute_pos(i1, j1);
	V2 = compute_pos(i2, j2);
	V3 = compute_pos(i3, j3);
	uv = compute_uv(V1, V2, V3);
	u = uv.x;
	v = uv.y;
	/* complex smooth coefficients
	cons++;
	Constraints.at<float>(cons, index_x(i2, j2)) = (1-u)*weight;
	Constraints.at<float>(cons, index_x(i3, j3)) = (u)*weight;
	Constraints.at<float>(cons, index_y(i3, j3)) = (v)*weight;
	Constraints.at<float>(cons, index_y(i2, j2)) = (-v)*weight;
	Constraints.at<float>(cons, index_x(i1, j1)) = -weight;
	cons++;
	Constraints.at<float>(cons, index_y(i2, j2)) = (1-u)*weight;
	Constraints.at<float>(cons, index_y(i3, j3)) = (u)*weight;
	Constraints.at<float>(cons, index_x(i2, j2)) = (v)*weight;
	Constraints.at<float>(cons, index_x(i3, j3)) = (-v)*weight;
	Constraints.at<float>(cons, index_y(i1, j1)) = -weight;
	*/

	///* simple smooth coefficients
	float s = norm(V1 - V2)/norm(V2 - V3);
	cons++;
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_x(i2, j2);
	SmoothVal[I] = -weight;
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_y(i3, j3);
	SmoothVal[I] = (-s)*weight;	
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_y(i2, j2);
	SmoothVal[I] = (s)*weight;	
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_x(i1, j1);
	SmoothVal[I] = weight;	

	cons++;
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_y(i2, j2);
	SmoothVal[I] = -weight;
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_x(i2, j2);
	SmoothVal[I] = (-s)*weight;	
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_x(i3, j3);
	SmoothVal[I] = (s)*weight;	
	I++;
	SmoothRow[I] = cons;
	SmoothCol[I] = index_y(i1, j1);
	SmoothVal[I] = weight;
	//*/
}

// the triangles
void asapWarp::addSmoothCoefficient(int & cons, int i1, int j1, int i2, int j2, int i3, int j3, float weight)
{
	Point2f V1, V2, V3, uv;
	float u, v;
	V1 = compute_pos(i1, j1);
	V2 = compute_pos(i2, j2);
	V3 = compute_pos(i3, j3);
	/* complex smooth coefficients
	uv = compute_uv(V1, V2, V3);
	u = uv.x;
	v = uv.y;
	
	cons++;
	Constraints.at<float>(cons, index_x(i2, j2)) = (1-u)*weight;
	Constraints.at<float>(cons, index_x(i3, j3)) = (u)*weight;
	Constraints.at<float>(cons, index_y(i3, j3)) = (v)*weight;
	Constraints.at<float>(cons, index_y(i2, j2)) = (-v)*weight;
	Constraints.at<float>(cons, index_x(i1, j1)) = -weight;
	cons++;
	Constraints.at<float>(cons, index_y(i2, j2)) = (1-u)*weight;
	Constraints.at<float>(cons, index_y(i3, j3)) = (u)*weight;
	Constraints.at<float>(cons, index_x(i2, j2)) = (v)*weight;
	Constraints.at<float>(cons, index_x(i3, j3)) = (-v)*weight;
	Constraints.at<float>(cons, index_y(i1, j1)) = -weight;
	*/

	///* simple smooth coefficients
	float s = norm(V1 - V2)/norm(V2 - V3);
	cons++;
	Constraints.at<float>(cons, index_x(i2, j2)) = -weight; 		// V2.x
	Constraints.at<float>(cons, index_y(i3, j3)) = (-s)*weight;		// V3.y
	Constraints.at<float>(cons, index_y(i2, j2)) = (s)*weight;		// V2.y
	Constraints.at<float>(cons, index_x(i1, j1)) = weight;			// V1.x
	cons++;
	Constraints.at<float>(cons, index_y(i2, j2)) = -weight;			// V2.y
	Constraints.at<float>(cons, index_x(i2, j2)) = (-s)*weight;		// V2.x
	Constraints.at<float>(cons, index_x(i3, j3)) = (s)*weight;		// V3.x
	Constraints.at<float>(cons, index_y(i1, j1)) = weight;			// V1.y
	//*/
}

void asapWarp::addDataCoefficientSparse(int & cons, int & I, Point2f prev_pt, Point2f now_pt)
{
	float x = prev_pt.x;
	float y = prev_pt.y;
	int ind_x = int(x) / quadWidth;
	int ind_y = int(y) / quadHeight;
	
	if (ind_x < width-1 && ind_y < height-1)
	{
		Point2f V00 = compute_pos(ind_x, ind_y);
		Point2f V11 = compute_pos(ind_x+1, ind_y+1);

		float u = (x - V00.x) / (V11.x - V00.x);
		float v = (y - V00.y) / (V11.y - V00.y);

		cons++;
		I++;
		ARow[I] = cons;
		ACol[I] = index_x(ind_x, ind_y);
		AVal[I] = (1-u)*(1-v);
		I++;
		ARow[I] = cons;
		ACol[I] = index_x(ind_x, ind_y+1);
		AVal[I] = (1-u)*v;
		I++;
		ARow[I] = cons;
		ACol[I] = index_x(ind_x+1, ind_y);
		AVal[I] = u*(1-v);
		I++;
		ARow[I] = cons;
		ACol[I] = index_x(ind_x+1, ind_y+1);
		AVal[I] = u*v;
		B[cons] = now_pt.x - prev_pt.x;

		cons++;
		I++;
		ARow[I] = cons;
		ACol[I] = index_y(ind_x, ind_y);
		AVal[I] = (1-u)*(1-v);
		I++;
		ARow[I] = cons;
		ACol[I] = index_y(ind_x, ind_y+1);
		AVal[I] = (1-u)*v;
		I++;
		ARow[I] = cons;
		ACol[I] = index_y(ind_x+1, ind_y);
		AVal[I] = u*(1-v);
		I++;
		ARow[I] = cons;
		ACol[I] = index_y(ind_x+1, ind_y+1);
		AVal[I] = u*v;
		B[cons] = now_pt.y - prev_pt.y;
	}
}

void asapWarp::addDataCoefficient(int & cons, Point2f prev_pt, Point2f now_pt)
{
	float x = prev_pt.x;
	float y = prev_pt.y;
	int ind_x = int(x) / quadWidth;
	int ind_y = int(y) / quadHeight;
	
	if (ind_x < width-1 && ind_y < height-1)
	{
		Point2f V00 = compute_pos(ind_x, ind_y);
		Point2f V11 = compute_pos(ind_x+1, ind_y+1);

		float u = (x - V00.x) / (V11.x - V00.x);
		float v = (y - V00.y) / (V11.y - V00.y);
	
		cons++;
		Constraints.at<float>(cons, index_x(ind_x, ind_y))     = (1-u)*(1-v);
		Constraints.at<float>(cons, index_x(ind_x, ind_y+1))   = (1-u)*v;
		Constraints.at<float>(cons, index_x(ind_x+1, ind_y))   = u*(1-v);
		Constraints.at<float>(cons, index_x(ind_x+1, ind_y+1)) = u*v;
		Constants.at<float>(cons, 0) 							= now_pt.x - prev_pt.x;
		//Constants.at<float>(cons, 0) 							= now_pt.x;
		cons++;
		Constraints.at<float>(cons, index_y(ind_x, ind_y))     = (1-u)*(1-v);
		Constraints.at<float>(cons, index_y(ind_x, ind_y+1))   = (1-u)*v;
		Constraints.at<float>(cons, index_y(ind_x+1, ind_y))   = u*(1-v);
		Constraints.at<float>(cons, index_y(ind_x+1, ind_y+1)) = u*v;
		Constants.at<float>(cons, 0) 							= now_pt.y - prev_pt.y;
		//Constants.at<float>(cons, 0) 							= now_pt.y;
	}
}

