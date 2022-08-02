
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "EyeRegions.hpp"
#include "CVFrame.hpp"
#include "HaarCascades.hpp"
#include "Landmarks.hpp"
#include "Known.hpp"
#include "Rect.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool qShowEyeWindows = false;
bool qUseEyeRefine = false;
bool qUseKnownEyes = false;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool FindVerticalSymmetry (cv::Rect& best, cv::Mat& MatOut, int& Ytop, int& Ybottom)
{
	const int MaxOffset = 6;

	// hack to make sure symmetry data is available at the bottom of the region
	best.height += MaxOffset;

	cv::Mat MatIn(greyscale_copy(best));

	std::clog << "Finding Vertical Symmetry" << std::endl;
	clogi("Inmat rows",MatIn.rows);
	clogi("Inmat cols",MatIn.cols);

	cv::equalizeHist(MatIn,MatIn);

	std::clog << "Create MatOut" << std::endl;
	MatOut.create(cv::Size(MatIn.cols,MatIn.rows),MatIn.type());
	std::clog << "Created MatOut" << std::endl;

	bool first = true;
	float Summin, Summax;
	float *Values = (float *)malloc(sizeof(float)*MatOut.cols*MatOut.rows);

	int Vindex = 0;

	for (int C = 0;  C < MatOut.cols;  ++C)
	{
		for (int R = 0;  R < MatOut.rows;  ++R)
		{
			float Sum = 0.0;

			int Doffsets = std::min(std::min(R,MatOut.rows-1-R),MaxOffset);

			for (int D = 1;  D <= Doffsets;  ++D)
			{
				int Vup = MatIn.at<uchar>(R-D,C);
				int Vdn = MatIn.at<uchar>(R+D,C);
				int Vvv = 255-3*abs(Vup-Vdn);
				Sum += Vvv;
			}

			if (first) { Summin = Summax = Sum;  first = false; }
			else { if (Sum < Summin) Summin = Sum;  if (Sum > Summax)  Summax = Sum; }

			Values[Vindex++] = Sum;
		}
	}

	Vindex = 0;

	for (int C = 0;  C < MatOut.cols;  ++C)
	{
		for (int R = 0;  R < MatOut.rows;  ++R)
		{
			float Sum = (Values[Vindex++]-Summin)*(255.0/(Summax-Summin));
			int V = std::max(0,std::min((int)Sum,255));
			MatOut.at<uchar>(R,C) = V;
		}
	}

	free(Values);

	// looking at the output from this, I don't think there's anything useful there, but try fitting a line to it just in case

	cv::equalizeHist(MatOut,MatIn);

	const double Rho = 1;
	const double Theta = CV_PI/180.0;
	int RequiredLineLength = MatIn.cols;
	int RequiredThreshold = RequiredLineLength/2;
	const int MaxGap = 3;

	std::vector<cv::Vec4i> good_lines;

	for ( ; ; )
	{
		good_lines.clear();

		std::vector<cv::Vec4i> lines;
	    cv::HoughLinesP(MatIn, lines, Rho, Theta, RequiredThreshold, RequiredLineLength, MaxGap);
	    clogi("Hough found lines: ",lines.size());

		for( size_t i = 0; i < lines.size(); i++ )
	    {
			cv::Point2i P0(lines[i][0], lines[i][1]);
			cv::Point2i P1(lines[i][2], lines[i][3]);
	        //line(MatOut, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,255,255), 3, 8 );

			if (abs(P0.x-P1.x) > abs(P0.y-P1.y))   // eye centres are horizontal, not vertical
			{
				if ((P0.y > MatOut.rows/4) && (P1.y > MatOut.rows/4))  // if the line strays into the top quarter of the region, it's wrong
				{
					good_lines.push_back(lines[i]);
				}
			}
	    }

        if (good_lines.size() > 0)
        {
			std::clog << "Hough transfer found " << std::to_string(good_lines.size()) << " at Threshold " << std::to_string(RequiredThreshold) << std::endl;

			for( size_t i = 0; i < good_lines.size(); i++ )
		    {
				cv::Point2i P0(good_lines[i][0], good_lines[i][1]);
				cv::Point2i P1(good_lines[i][2], good_lines[i][3]);
				clog2i("Hough found good line from ",P0);
				clog2i("Hough found good line to ",P1);

		        //line(MatOut, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,255,255), 3, 8 );
		    }

			cv::Point2i P0(good_lines[0][0], good_lines[0][1]);
			cv::Point2i P1(good_lines[0][2], good_lines[0][3]);
			Ytop = std::min(P0.y,P1.y);
			Ybottom = std::max(P0.y,P1.y);
			MatIn.release();
			return true;
		}

        RequiredThreshold--;
        RequiredLineLength -= 2;

        if (RequiredLineLength <= MaxGap)
        {
			std::clog << "Nothing there to be found" << std::endl;
			MatIn.release();
			return false;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void AdjustRect (cv::Rect& best, int Ytop, int Ybottom)
{
	best.y += Ytop-6;
	best.height = (Ybottom-Ytop)+13;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void refine_eye_regions (FaceAndEyes& this_face)
{
	cv::Mat LEyeGrey;
	cv::Mat REyeGrey;

	if (false)
	{
		int Ytop,Ybottom;
		if (FindVerticalSymmetry(this_face.leye.best.box,LEyeGrey,Ytop,Ybottom))  AdjustRect(this_face.leye.best.box,Ytop,Ybottom);
		if (FindVerticalSymmetry(this_face.reye.best.box,REyeGrey,Ytop,Ybottom))  AdjustRect(this_face.reye.best.box,Ytop,Ybottom);
	}

	if (qShowEyeWindows)
	{
		cv::imshow( cLEyeWindowName, LEyeGrey);
		cv::imshow( cREyeWindowName, REyeGrey);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////

void estimate_initial_eye_regions (FaceAndEyes& this_face)
{
	find_eyes_using_haar_cascades(this_face);
	tidy_eyes_using_common_sense(this_face);
	set_eye_landmarks(this_face);
}

////////////////////////////////////////////////////////////////////////////////////////////

void estimate_improved_eye_regions (FaceAndEyes& this_face)
{
    if (qUseEyeRefine)  refine_eye_regions(this_face);
}

////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void use_known_eyes_if_available (FaceAndEyes& this_face)
{
	cv::Rect eye;
	if (is_known_Leye(eye))  this_face.leye.best.box = eye;
	if (is_known_Reye(eye))  this_face.reye.best.box = eye;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

