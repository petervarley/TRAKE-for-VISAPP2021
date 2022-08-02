
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "Types.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void clogi (const char *preamble, const int value)
{
	std::clog << preamble << " " << std::to_string(value) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clogf (const char *preamble, const float value)
{
	std::clog << preamble << " " << std::to_string(value) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clogproportion (const char *preamble, int a, int b)
{
	float af = a;
	float bf = b;
	std::clog << "Proportion " << preamble << " is " << std::to_string(af/bf) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clog2i (const char *preamble, const cv::Point2i& pixel)
{
	std::clog << preamble << " x=" << std::to_string(pixel.x) << " y=" << std::to_string(pixel.y) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clog2f (const char *preamble, const cv::Point2f& point)
{
	std::clog << preamble << " x=" << std::to_string(point.x) << " y=" << std::to_string(point.y) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clogrect (const char *preamble, const cv::Rect& rect)
{
	std::clog << preamble << " x=" << std::to_string(rect.x) << " y=" << std::to_string(rect.y) << " w=" << std::to_string(rect.width) << " h=" << std::to_string(rect.height) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clogregion (const char *preamble, const Region& region)
{
	std::clog << preamble << region.tag << " x=" << std::to_string(region.box.x) << " y=" << std::to_string(region.box.y) << " w=" << std::to_string(region.box.width) << " h=" << std::to_string(region.box.height) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void clogprediction (const char *preamble, const Prediction pred)
{
	std::clog << preamble << " x=" << std::to_string(pred.p.x) << " y=" << std::to_string(pred.p.y) << " weight=" << std::to_string(pred.w) << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
