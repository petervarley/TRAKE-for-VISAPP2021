#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <opencv2/core/core.hpp>

////////////////////////////////////////////////////////////////////////////////////////////
// Includes type definitions, logging functions for basic types, occasional global variables,
// and anything else which doesn't fit comfortably in one package

////////////////////////////////////////////////////////////////////////////////////////////

// Window names
constexpr char *cSourceWindowName = "Image";
constexpr char *cLEyeWindowName = "Left Eye";
constexpr char *cREyeWindowName = "Right Eye";
constexpr char *cNoseWindowName = "Nose";

////////////////////////////////////////////////////////////////////////////////////////////

// General-purpose type definitions

typedef struct
{
	cv::Rect box;
	char tag;
} Region;


typedef struct
{
	cv::Rect ROIRect;
	Region best;
	std::vector<Region>list;
	std::vector<Region>lis2;
} Feature;

enum goodness_type { nothing = 0, nose_only = 1, nose_and_mouth = 2, one_eye = 3, two_eyes = 4, two_eyes_exact = 5 };

typedef struct
{
	goodness_type goodness;
	bool process_this;
	cv::Rect face;
	float distance;
	Feature smile;
	Feature leye;
	Feature reye;
	Feature nose;
	cv::Point2i lpupil;
	cv::Point2i rpupil;
	cv::Point2i lnostril;
	cv::Point2i rnostril;
	cv::Rect gaze_prediction;
	cv::Point2i screen_prediction;
} FaceAndEyes;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct
{
	cv::Point2f p;
	float w;
} Prediction;

////////////////////////////////////////////////////////////////////////////////////////////

// General-purpose logging routines
void clogi (const char *preamble, const int value);
void clogf (const char *preamble, const float value);
void clogproportion (const char *preamble, int a, int b);
void clog2i (const char *preamble, const cv::Point2i& pixel);
void clog2f (const char *preamble, const cv::Point2f& point);
void clogrect (const char *preamble, const cv::Rect& rect);
void clogregion (const char *preamble, const Region& region);
void clogprediction (const char *preamble, const Prediction pred);

////////////////////////////////////////////////////////////////////////////////////////////

inline int roundfloat (float f) { return (int)(f+0.5f); }

////////////////////////////////////////////////////////////////////////////////////////////

#endif
