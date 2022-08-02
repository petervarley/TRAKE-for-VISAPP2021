
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "Noses.hpp"
#include "CVFrame.hpp"
#include "HaarCascades.hpp"
#include "FileLocations.hpp"
#include "Known.hpp"
#include "Rect.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool qShowNoseWindow = false;
bool qUseNoseRefine = true;
bool qUseKnownNoses = false;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the old version which used pixel intensities. It didn't work well.

static bool old_find_horizontal_symmetry (cv::Rect& best, cv::Mat& MatOut, int& Xleft, int& Xright)
{
	const int MaxOffset = 13;

	// hack to make sure symmetry data is available at the bottom of the region

	cv::Mat MatIn(greyscale_copy(best));

	cv::equalizeHist(MatIn,MatIn);

	//std::clog << "Create MatOut" << std::endl;
	MatOut.create(cv::Size(MatIn.cols,MatIn.rows),MatIn.type());
	//std::clog << "Created MatOut" << std::endl;

	bool first = true;
	float Summin, Summax;
	float *Values = (float *)malloc(sizeof(float)*MatOut.cols*MatOut.rows);

	int Vindex = 0;

	for (int R = 0;  R < MatOut.rows;  ++R)
	{
		for (int C = 0;  C < MatOut.cols;  ++C)
		{
			float Sum = 0.0;

			int Doffsets = std::min(std::min(C,MatOut.cols-1-C),MaxOffset);

			for (int D = 1;  D <= Doffsets;  ++D)
			{
				int Vup = MatIn.at<uchar>(R,C-D);
				int Vdn = MatIn.at<uchar>(R,C+D);
				int Vvv = 255-3*abs(Vup-Vdn);
				//clogi("Vup",Vup);
				//clogi("Vdn",Vdn);
				Sum += Vvv;
			}
			//clogi("Doffsets,",Doffsets);
			// don't bother "correcting" for the number of comparisons, borders should be dark
			//clogf("Sum",10000.0+Sum);

			if (first) { Summin = Summax = Sum;  first = false; }
			else { if (Sum < Summin) Summin = Sum;  if (Sum > Summax)  Summax = Sum; }

			Values[Vindex++] = Sum;
		}
	}
	//clogf("Summin",Summin);
	//clogf("Summax",Summax);

	Vindex = 0;

	for (int R = 0;  R < MatOut.rows;  ++R)
	{
		for (int C = 0;  C < MatOut.cols;  ++C)
		{
			float Sum = (Values[Vindex++]-Summin)*(255.0/(Summax-Summin));
			int V = std::max(0,std::min((int)Sum,255));
			MatOut.at<uchar>(R,C) = V;
		}
	}

	free(Values);

	cv::equalizeHist(MatOut,MatIn);

	const double Rho = 1;
	const double Theta = CV_PI/180.0;
	int RequiredLineLength = MatIn.rows;
	int RequiredThreshold = RequiredLineLength/2;
	const int MaxGap = 3;

	std::vector<cv::Vec4i> good_lines;

	for ( ; ; )
	{
		good_lines.clear();

		std::vector<cv::Vec4i> lines;
	    cv::HoughLinesP(MatIn, lines, Rho, Theta, RequiredThreshold, RequiredLineLength, MaxGap);
	    //clogi("Hough found lines: ",lines.size());

		for( size_t i = 0; i < lines.size(); i++ )
	    {
			cv::Point2i P0(lines[i][0], lines[i][1]);
			cv::Point2i P1(lines[i][2], lines[i][3]);
			//clog2i("Hough found line from ",P0);
			//clog2i("Hough found line to ",P1);
	        //line(MatOut, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,255,255), 3, 8 );

			if ((2*abs(P0.x-P1.x)) < abs(P0.y-P1.y))   // nose centres are vertical, not horizontal or diagonal
			{
				good_lines.push_back(lines[i]);
			}
	    }

        if (good_lines.size() > 0)
        {
			//std::clog << "Hough transfer found " << std::to_string(good_lines.size()) << " at Threshold " << std::to_string(RequiredThreshold) << std::endl;

			for( size_t i = 0; i < good_lines.size(); i++ )
		    {
				cv::Point2i P0(good_lines[i][0], good_lines[i][1]);
				cv::Point2i P1(good_lines[i][2], good_lines[i][3]);
				//clog2i("Hough found good line from ",P0);
				//clog2i("Hough found good line to ",P1);

		        //line(MatOut, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,255,255), 3, 8 );
		    }

			cv::Point2i P0(good_lines[0][0], good_lines[0][1]);
			cv::Point2i P1(good_lines[0][2], good_lines[0][3]);
			Xleft = std::min(P0.x,P1.x);
			Xright = std::max(P0.x,P1.x);
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

	// std::clog << "Found Vertical Symmetry" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the new version which uses intensity gradients. It is even worse than the old one.

static bool find_horizontal_symmetry (cv::Rect& best, cv::Mat& MatOut, int& Xleft, int& Xright)
{
	const int MaxOffset = 13;

	// hack to make sure symmetry data is available at the bottom of the region

	cv::Mat MatIn(greyscale_copy(best));

	cv::equalizeHist(MatIn,MatIn);

	//std::clog << "Create MatOut" << std::endl;
	MatOut.create(cv::Size(MatIn.cols,MatIn.rows),MatIn.type());
	//std::clog << "Created MatOut" << std::endl;

	bool first = true;
	float Summin, Summax;
	float *Values = (float *)malloc(sizeof(float)*MatOut.cols*MatOut.rows);

	int Vindex = 0;

	for (int R = 0;  R < MatOut.rows;  ++R)
	{
		for (int C = 0;  C < MatOut.cols;  ++C)
		{
			float Sum = 0.0;

			int Doffsets = std::min(std::min(C-1,MatOut.cols-2-C),MaxOffset);

			for (int D = 0;  D <= Doffsets;  ++D)
			{
				int Vl0 = MatIn.at<uchar>(R,C-D);
				int Vl1 = MatIn.at<uchar>(R,C-D-1);
				int Vr0 = MatIn.at<uchar>(R,C+D);
				int Vr1 = MatIn.at<uchar>(R,C+D+1);

				int Vl = Vl0-Vl1;
				int Vr = Vr0-Vr1;
				int Vvv = Vl*Vr;
				//clogi("Vvv",Vvv);
				Sum += Vvv;
			}
			// don't bother "correcting" for the number of comparisons, borders should be dark

			if (first) { Summin = Summax = Sum;  first = false; }
			else { if (Sum < Summin) Summin = Sum;  if (Sum > Summax)  Summax = Sum; }

			Values[Vindex++] = Sum;
		}
	}

	// Preliminary test results: Sum is in the range -31073..28284
	// The 5th and 95th percentiles are -2944..4232, and that's what I shall scale the output matrix to

	Vindex = 0;

	for (int R = 0;  R < MatOut.rows;  ++R)
	{
		for (int C = 0;  C < MatOut.cols;  ++C)
		{
			float Sum = (Values[Vindex++]+2944)*(255.0/(4232+2944));
			int V = std::max(0,std::min((int)Sum,255));
			MatOut.at<uchar>(R,C) = V;
		}
	}

	free(Values);

	cv::equalizeHist(MatOut,MatIn);

	const double Rho = 1;
	const double Theta = CV_PI/180.0;
	int RequiredLineLength = MatIn.rows;
	int RequiredThreshold = RequiredLineLength/2;
	const int MaxGap = 3;

	std::vector<cv::Vec4i> good_lines;

	for ( ; ; )
	{
		good_lines.clear();

		std::vector<cv::Vec4i> lines;
	    cv::HoughLinesP(MatIn, lines, Rho, Theta, RequiredThreshold, RequiredLineLength, MaxGap);
	    //clogi("Hough found lines: ",lines.size());

		for( size_t i = 0; i < lines.size(); i++ )
	    {
			cv::Point2i P0(lines[i][0], lines[i][1]);
			cv::Point2i P1(lines[i][2], lines[i][3]);

			if ((2*abs(P0.x-P1.x)) < abs(P0.y-P1.y))   // nose centres are vertical, not horizontal or diagonal
			{
				good_lines.push_back(lines[i]);
			}
	    }

        if (good_lines.size() > 0)
        {
			//std::clog << "Hough transfer found " << std::to_string(good_lines.size()) << " at Threshold " << std::to_string(RequiredThreshold) << std::endl;

			for( size_t i = 0; i < good_lines.size(); i++ )
		    {
				cv::Point2i P0(good_lines[i][0], good_lines[i][1]);
				cv::Point2i P1(good_lines[i][2], good_lines[i][3]);
		    }

			cv::Point2i P0(good_lines[0][0], good_lines[0][1]);
			cv::Point2i P1(good_lines[0][2], good_lines[0][3]);
			Xleft = std::min(P0.x,P1.x);
			Xright = std::max(P0.x,P1.x);
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

static void find_nosetip_using_centre_of_region (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	this_nosetip.p.x = mxf(this_face.nose.best.box);
	this_nosetip.p.y = myf(this_face.nose.best.box);
	this_nosetip.w = 0.568;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSEC",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void find_nosetip_using_brightest_boxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 9;

	int best_so_far = 0;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					total += px;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	// In testing, it was found that the y prediction is 5 or 6 too low.
	// The implication is that the prediction is 5 or even 6 pixels too high,
	// and the brightest spot on the nose is not the tip but the slope 5 or 6 pixels above it.
	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2 + 6.8;
	this_nosetip.w = 0.333;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSEB",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void find_nose_tip_using_hilo_double_boxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 7;

	int best_so_far = -1;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					int qx = greyscale_copy.at<uchar>(R+rr+kBoxNose,C+cc);
					total += px-qx;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2 + 2.28;
	this_nosetip.w = 0.4004;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSED",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void FindNoseTipUsingVerticalTripleBoxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 7;

	int best_so_far = -1;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					int qx = greyscale_copy.at<uchar>(R+rr-kBoxNose,C+cc);
					int rx = greyscale_copy.at<uchar>(R+rr+kBoxNose,C+cc);
					total += px+px-qx-rx;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2 + 2.93;
	this_nosetip.w = 0.281;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSEV",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void FindNoseTipUsingHorizontalTripleBoxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 9;

	int best_so_far = -1;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					int qx = greyscale_copy.at<uchar>(R+rr,C+cc-kBoxNose);
					int rx = greyscale_copy.at<uchar>(R+rr,C+cc+kBoxNose);
					total += px+px-qx-rx;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2;
	this_nosetip.w = 0.260;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSEH",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void FindNoseTipUsingTriangularTripleBoxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 7;
	const int Knostrilsx = 4;
	const int Knostrilsy = 3;

	int best_so_far = -1;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					int qx = greyscale_copy.at<uchar>(R+rr+Knostrilsy,C+cc-Knostrilsx);
					int rx = greyscale_copy.at<uchar>(R+rr+Knostrilsy,C+cc+Knostrilsx);
					total += px+px-qx-rx;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2 - 0.06;
	this_nosetip.w = 0.645;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSET",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void FindNoseTipUsingUptriangularTripleBoxcar (FaceAndEyes& this_face, Feature& this_nose, Prediction& this_nosetip)
{
	const int kBoxNose = 11;
	const int Knostrilsx = 9;
	const int Knostrilsy = 7;

	int best_so_far = -1;
	int bestR;
	int bestC;

	for (int R0 = 0;  R0 < this_nose.best.box.height-(kBoxNose-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(kBoxNose-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kBoxNose;  ++rr)
			{
				for (int cc = 0;  cc < kBoxNose;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					int qx = greyscale_copy.at<uchar>(R+rr-Knostrilsy,C+cc-Knostrilsx);
					int rx = greyscale_copy.at<uchar>(R+rr-Knostrilsy,C+cc+Knostrilsx);
					total += px+px-qx-rx;
				}
			}

			if (total > best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	this_nosetip.p.x = bestC+(kBoxNose-1)/2;
	this_nosetip.p.y = bestR+(kBoxNose-1)/2 + 5.77;
	this_nosetip.w = 0.313;

	cv::Point2i known_nosetip;

	if (is_known_nose(known_nosetip))
	{
		log_prediction_results("NOSEU",known_nosetip,this_nosetip.p);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void find_nostrils_using_boxcar (FaceAndEyes& this_face, Feature& this_nose, cv::Point2i& this_nostril, const cv::Point2i& previous_nostril)
{
	int best_so_far = 0;
	const int startR = this_nose.best.box.height/2;
	int bestR;
	int bestC;

	for (int R0 = startR;  R0 < this_nose.best.box.height-(3-1);  ++R0)
	{
		int R = R0 + this_nose.best.box.y;

		for (int C0 = 0;  C0 < this_nose.best.box.width-(5-1);  ++C0)
		{
			int C = C0 + this_nose.best.box.x;

			if ((abs((C+2)-previous_nostril.x)>=10) || (abs((R+1)-previous_nostril.y)>=8))
			{
				int total = 0;

				for (int rr = 0;  rr < 3;  ++rr)
				{
					for (int cc = 0;  cc < 5;  ++cc)
					{
						int px = 255-greyscale_copy.at<uchar>(R+rr,C+cc);
						total += px;
					}
				}

				if (total > best_so_far)
				{
					best_so_far = total;
					bestR = R;
					bestC = C;
				}
			}
		}
	}

	this_nostril.x = bestC+2;
	this_nostril.y = bestR+1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void find_weighted_mean (int Count, Prediction *Predictions, cv::Point2f& Mean)
{
	Mean.x = Mean.y = 0.0;
	float Total = 0.0;

	for (int i=0; i<Count; ++i)
	{
		Mean.x += Predictions[i].w*Predictions[i].p.x;
		Mean.y += Predictions[i].w*Predictions[i].p.y;
		Total += Predictions[i].w;
	}

	Mean.x /= Total;
	Mean.y /= Total;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static float distance_multiplier (const cv::Point2f& A, const cv::Point2f& B)
{
	float dx = A.x-B.x;
	float dy = A.y-B.y;
	float dh = dx*dx+dy*dy;
	return (dh <= 1.0) ? 1.0 : 1.0/dh;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void weight_predictions_by_distance_from_mean (int Count, Prediction *Predictions, cv::Point2f& Mean)
{
	for (int i=0; i<Count; ++i)  Predictions[i].w *= distance_multiplier(Predictions[i].p,Mean);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_NUMBER_OF_FERNS  6

char *nose_fern_list = "CT";

void refine_nose_tip (FaceAndEyes& this_face)
{
	cv::Mat NoseGrey;
	blur_and_equalise(greyscale_copy(this_face.nose.best.box),NoseGrey);

	char MessageBuffer[32];
	sprintf(MessageBuffer,"REFINE NOSE %c",this_face.nose.best.tag);

	if (qUseNoseRefine)
	{
		Prediction Predictions[MAX_NUMBER_OF_FERNS];
		int number_of_ferns = strlen(nose_fern_list);

		for (int n=0; n<number_of_ferns; ++n)
		{
			switch (nose_fern_list[n])
			{
				case 'C':
					find_nosetip_using_centre_of_region(this_face,this_face.nose,Predictions[n]);
				break;

				case 'T':
					FindNoseTipUsingTriangularTripleBoxcar(this_face,this_face.nose,Predictions[n]);
				break;

				case 'U':
					FindNoseTipUsingUptriangularTripleBoxcar(this_face,this_face.nose,Predictions[n]);
				break;

				case 'V':
					FindNoseTipUsingVerticalTripleBoxcar(this_face,this_face.nose,Predictions[n]);
				break;

				case 'H':
					FindNoseTipUsingHorizontalTripleBoxcar(this_face,this_face.nose,Predictions[n]);
				break;

				case 'D':
					find_nose_tip_using_hilo_double_boxcar(this_face,this_face.nose,Predictions[n]);
				break;

				case 'B':
					find_nosetip_using_brightest_boxcar(this_face,this_face.nose,Predictions[n]);
					// is brightness bimodal?
					// earlier experiments have shown no advantage to adding an extra prediction at y+5.5
				break;
			}
			clogprediction("NOSE prediction ",Predictions[n]);
		}

		cv::Point2f Mean;
		find_weighted_mean(number_of_ferns,Predictions,Mean);

		if (number_of_ferns > 2)
		{
			weight_predictions_by_distance_from_mean(number_of_ferns,Predictions,Mean);
			find_weighted_mean(number_of_ferns,Predictions,Mean);
		}

		cv::Point2i noseTip(roundfloat(Mean.x),roundfloat(Mean.y));
		this_face.nose.best.box.x = noseTip.x-3;
		this_face.nose.best.box.width = 7;
		this_face.nose.best.box.y = noseTip.y-2;
		this_face.nose.best.box.height = 5;
		this_face.nose.best.tag = '=';

		cv::Point2i known_nosetip;
		if (is_known_nose(known_nosetip))  log_prediction_results(MessageBuffer,known_nosetip,noseTip);
	}
	else
	{
		std::clog << "NOT USING NOSE REFINE" << std::endl;
	}

	if (qShowNoseWindow)
	{
		cv::imshow( cNoseWindowName, NoseGrey);

		if (qSaveFrames)
		{
			char output_filename[256];
			sprintf(output_filename,"%s/%s",flocOutput2,known_facename);
			write_frame_to_file(output_filename,NoseGrey);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////
// Note that initial nose estimation requires initial eye and mouth estimation and must follow them

void estimate_initial_noses (FaceAndEyes& this_face)
{
	find_nose_using_haar_cascades(this_face);

	if (qWhichNoseCascade == 2)
	{
		for (int n=0; n<this_face.nose.lis2.size(); ++n)  this_face.nose.list.push_back(this_face.nose.lis2[n]);
	}

	tidy_noses_using_common_sense(this_face);
	set_nose_landmark(this_face);
}

////////////////////////////////////////////////////////////////////////////////////////////

void estimate_improved_noses (FaceAndEyes& this_face)
{
    refine_nose_tip(this_face);
}

////////////////////////////////////////////////////////////////////////////////////////////

void locate_accurate_noses (FaceAndEyes& this_face)
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void use_known_nose_if_available (FaceAndEyes& this_face)
{
	cv::Rect nose;
	cv::Point2i tip;
	if (is_known_nose(tip))
	{
		nose.x = tip.x-3;
		nose.y = tip.y-3;
		nose.width = nose.height = 7;
		this_face.nose.best.box = nose;
		this_face.nose.best.tag = 'K';

		// analytical code, shouldn't be here, but where else to put it?
		cv::Rect face;
		if (is_known_face(face))  clogf ("Known Nose Location down from top of Face ",(float)(tip.y-face.y)/(float)(face.height));
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

