
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "Pupils.hpp"
#include "CVFrame.hpp"
#include "Known.hpp"
#include "Rect.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

int kPupilBoxcarSize = 7;
bool qUseLower60 = true;

bool qUseKnownPupils = false;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void find_one_pupil_using_boxcar (FaceAndEyes& this_face, Feature& this_eye, cv::Point2i& this_pupil, char which)
{
	int best_so_far = 32000;
	int bestR;
	int bestC;
	int startR = 0;
	if (qUseLower60) startR += this_eye.best.box.height*2/5;

	for (int R0 = startR;  R0 < this_eye.best.box.height-(kPupilBoxcarSize-1);  ++R0)
	{
		int R = R0 + this_eye.best.box.y;

		for (int C0 = 0;  C0 < this_eye.best.box.width-(kPupilBoxcarSize-1);  ++C0)
		{
			int C = C0 + this_eye.best.box.x;
			int total = 0;

			for (int rr = 0;  rr < kPupilBoxcarSize;  ++rr)
			{
				for (int cc = 0;  cc < kPupilBoxcarSize;  ++cc)
				{
					int px = greyscale_copy.at<uchar>(R+rr,C+cc);
					total += px;
				}
			}

			//std::clog << "Total at R=" << std::to_string(R) << ", C=" << std::to_string(C) << " is " << std::to_string(total) << std::endl;

			if (total < best_so_far)
			{
				best_so_far = total;
				bestR = R;
				bestC = C;
			}
		}
	}

	// std::clog << "Total at R=" << std::to_string(bestR) << ", C=" << std::to_string(bestC) << " is " << std::to_string(best_so_far) << std::endl;
	this_pupil.x = bestC+kPupilBoxcarSize/2;
	this_pupil.y = bestR+kPupilBoxcarSize/2;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Requires that best leye and best reye are already set on input

void estimate_initial_pupils (FaceAndEyes& this_face)
{
	find_one_pupil_using_boxcar(this_face,this_face.leye,this_face.lpupil,'L');
	find_one_pupil_using_boxcar(this_face,this_face.reye,this_face.rpupil,'R');
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void use_known_pupils_if_available (FaceAndEyes& this_face)
{
	cv::Point2i pupil;
	if (is_known_Lpupil(pupil))  this_face.lpupil = pupil;
	if (is_known_Rpupil(pupil))  this_face.rpupil = pupil;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

