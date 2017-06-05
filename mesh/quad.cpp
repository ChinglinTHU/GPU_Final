#include "quad.h"

#include <cmath>

using namespace std;
using namespace cv;

Quad::Quad(Point v00, Point v01, Point v10, Point v11)
{
	V00 = v00;
	V01 = v01;
	V10 = v10;
	V11 = v11;
}

Quad::~Quad(){}

bool Quad::isPointIn(Point pt)
{
	return isPointInTriangular(pt, V00, V01, V11) || isPointInTriangular(pt, V00, V10, V11);
}

bool Quad::isPointsIn(vector<Point> pts)
{
	for(int i = 0; i < pts.size(); i++)
	{
		if(isPointIn(pts[i]))
			return true;
	}
	return false;
}

bool Quad::getBilinearCoordinates(Point pt, float* coefficients)
{
	float a_x = V00.x - V01.x - V10.x + V11.x;
	float b_x = -V00.x + V01.x;
    float c_x = -V00.x + V10.x;
    float d_x = V00.x - pt.x;
            
    float a_y = obj.V00.y - obj.V01.y - obj.V10.y + obj.V11.y;
    float b_y = -V00.y + V01.y;
    float c_y = -V00.y + V10.y;
    float d_y = V00.y - pt.y;
     
    float bigA = -a_y*b_x + b_y*a_x;
    float bigB = -a_y*d_x - c_y*b_x + d_y*a_x +b_y*c_x;
    float bigC = -c_y*d_x + d_y*c_x;
    
    float tmp1 = -1;
    float tmp2 = -1;
    float tmp3 = -1;
    float tmp4 = -1;

    if (bigB*bigB - 4*bigA*bigC >= 0.0)
    {
        if (abs(bigA) >= 0.000001)
        {
            tmp1 = ( -bigB + sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
            tmp2 = ( -bigB - sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
        }
        else
            tmp1 = -bigC/bigB;
        
        if ( tmp1 >= -0.999999 && tmp1 <= 1.000001)
        {
            tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
            tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
            if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
            {
                k1 = tmp1;
                k2 = tmp3;
            }
            else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
            {
                k1 = tmp1;
                k2 = tmp4;
            }
        }
        if ( tmp2 >= -0.999999 && tmp2 <= 1.000001)
        {    
            if (tmp3 >= -0.999999 && tmp3 <= 1.000001)
            {    
            	k1 = tmp2;
                k2 = tmp3;
            }
            else if (tmp4 >= -0.999999 && tmp4 <= 1.000001)
            {
                k1 = tmp2;
                k2 = tmp4;
            }
        }
	}

    if (k1>=-0.999999 && k1<=1.000001 && k2>=-0.999999 && k2<=1.000001)
    {    
        float coe1 = (1.0-k1)*(1.0-k2);
        float coe2 = k1*(1.0-k2);
        float coe3 = (1.0-k1)*k2;
        float coe4 = k1*k2;
        
        coefficients[0] = coe1;
        coefficients[1] = coe2;
        coefficients[2] = coe3;
        coefficients[3] = coe4;
        
        return true;
    }
    else
        return false;
}

float Quad::getMinX()
{
	float minx = min(V00.x, V01.x);
	minx = min(minx, V10.x);
	minx = min(minx, V11.x);
	return minx;
}

float Quad::getMaxX()
{
	float maxx = min(V00.x, V01.x);
	maxx = min(maxx, V10.x);
	maxx = min(maxx, V11.x);
	return maxx;
}

float Quad::getMinY()
{
	float miny = min(V00.y, V01.y);
	miny = min(miny, V10.y);
	miny = min(miny, V11.y);
	return miny;
}

float Quad::getMaxY()
{
	float maxy = min(V00.y, V01.y);
	maxy = min(maxy, V10.y);
	maxy = min(maxy, V11.y);
	return maxy;
}

bool Quad::isPointInTriangular(Point pt, Point V0, Point V1, Point V2)
{
    float lambda1 = ((V1.y-V2.y)*(pt.x-V2.x) + (V2.x-V1.x)*(pt.y-V2.y)) / ((V1.y-V2.y)*(V0.x-V2.x) + (V2.x-V1.x)*(V0.y-V2.y));
	float lambda2 = ((V2.y-V0.y)*(pt.x-V2.x) + (V0.x-V2.x)*(pt.y-V2.y)) / ((V2.y-V0.y)*(V1.x-V2.x) + (V0.x-V2.x)*(V1.y-V2.y));
    float lambda3 = 1-lambda1-lambda2;
    if (lambda1 >= 0.0 && lambda1 <= 1.0 && lambda2 >= 0.0 && lambda2 <= 1.0 && lambda3 >= 0.0 && lambda3 <= 1.0)
        return true;
    else
        return false;
}

// bool isPointsInTriangular(std::vector<cv::Point> pts, cv::Point v0, cv::Point v1, cv::Point v2);