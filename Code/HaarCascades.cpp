
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "HaarCascades.hpp"
#include "CVFrame.hpp"
#include "Landmarks.hpp"
#include "Rect.hpp"
#include "Constants.hpp"
#include "FileLocations.hpp"

static cv::CascadeClassifier face_cascade;
static cv::CascadeClassifier fac2_cascade;

static cv::CascadeClassifier smile_cascade;
static cv::CascadeClassifier mouth_cascade;

static cv::CascadeClassifier nose_cascade;
static cv::CascadeClassifier nos2_cascade;

static cv::CascadeClassifier eye1_cascade;
static cv::CascadeClassifier eye2_cascade;
static int eye_cascade_count = 2;

///////////////////////////////////////////////////////////////////////////////////////////////

bool qUseCascades = true;
bool qHaveCascades = false;

///////////////////////////////////////////////////////////////////////////////////////////////

int qWhichFaceCascade = 2;
int qWhichEyeCascade  = 2;
int qWhichNoseCascade = 2;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// A typical face region in a 250x250 image might be 128x128
// Anything much larger than that isn't a face, and anything much smaller than that is too small to be useful.
// Set up the thresholds so that 200x200 is the maximum and 80x80 is the minimum
//
// However, in the real world, we want to deal with larger images in which the face is just a small part.
// So perhaps it is better to hard-code the thresholds rather than set them according to the image size.
//
// At 2m (the minimum), the face size could be anywhere between 70 and 175 pixels.
// At 10m (the maximum), the face size could be anywhere between 14 and 35 pixels.
// So the size of the face could be anywhere between 14 and 175 pixels. That's a wide range which could lead to large numbers of false positives.
//
// All of which is redundant if we are in a video sequence and have a previous frame to guide us
//
// This routine should be modified so that it can be called several times for video sequences with more than one face

static void internal_find_face_using_haar_cascades (void)
{
	int MinThreshold;
	int MaxThreshold;
	cv::Rect faceROI;

	MinThreshold = 14;   // see above
	MaxThreshold = 175;  // see above
	faceROI.x = faceROI.y = 0;
	faceROI.width = greyscale_copy.cols;
	faceROI.height = greyscale_copy.rows;

	std::vector<cv::Rect> faces_list;
	faces_list.clear();

   	face_cascade.detectMultiScale( greyscale_copy(faceROI), faces_list, 1.1, 2, 0, cv::Size(MinThreshold,MinThreshold), cv::Size(MaxThreshold,MaxThreshold) );

   	if (faces_list.size() == 0)
   	{
		// never mind, there's a backup
	   	fac2_cascade.detectMultiScale( greyscale_copy(faceROI), faces_list, 1.1, 2, 0, cv::Size(MinThreshold,MinThreshold), cv::Size(MaxThreshold,MaxThreshold) );
	}

    for (faces_and_eyes_count = 0;  faces_and_eyes_count < faces_list.size(); ++faces_and_eyes_count)
    {
		faces_and_eyes_list[faces_and_eyes_count].face.x = faces_list[faces_and_eyes_count].x + faceROI.x;
		faces_and_eyes_list[faces_and_eyes_count].face.y = faces_list[faces_and_eyes_count].y + faceROI.y;
		faces_and_eyes_list[faces_and_eyes_count].face.width = faces_list[faces_and_eyes_count].width;
		faces_and_eyes_list[faces_and_eyes_count].face.height = faces_list[faces_and_eyes_count].height;
	}
}

//-----------------------------------------------------------------------------------------------------------

void find_face_using_haar_cascades (void)
{
	// shrink large images so that they don't take so long
	int Dimensions = working_frame.rows+working_frame.cols;
	if (Dimensions > 1000)
	{
		float shrink = ((float)Dimensions)/1000.0;
		cv::resize(working_frame,shrunk_frame,cv::Size(),1.0d/shrink,1.0d/shrink,cv::INTER_AREA);

		create_shrunk_greyscale_copy();
		internal_find_face_using_haar_cascades();

	    for (int f = 0;  f < faces_and_eyes_count; ++f)
	    {
			faces_and_eyes_list[f].face.x *= shrink;
			faces_and_eyes_list[f].face.y *= shrink;
			faces_and_eyes_list[f].face.width *= shrink;
			faces_and_eyes_list[f].face.height *= shrink;
		}

		// put things back the way they should be
		create_greyscale_copy();
	}
	else
	{
		// for reasonably-sized images
		internal_find_face_using_haar_cascades();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

static void convert_rect_to_region (const cv::Rect& rect,  const cv::Rect& roi,  const char tag,  Region& region)
{
	region.box.x = rect.x + roi.x;
	region.box.y = rect.y + roi.y;
	region.box.width = rect.width;
	region.box.height = rect.height;
	region.tag = tag;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

static void run_cascade_and_store (cv::CascadeClassifier& classifier, cv::Rect& roi, const cv::Size minsize, const cv::Size maxsize, const char tag, std::vector<Region> &regions)
{
    std::vector<cv::Rect> rectvector;
    classifier.detectMultiScale(greyscale_copy(roi), rectvector, 1.1, 2, 0, minsize, maxsize);

	regions.resize(rectvector.size());
	for (size_t j = 0; j < rectvector.size(); j++)  convert_rect_to_region(rectvector[j],roi,tag,regions[j]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// A typical mouth region in a 250x250 image might be 50x25
// Anything much larger than that isn't a mouth, and anything much smaller than that is too small to be useful.
// Set up the thresholds so that 80x80 is the maximum and 15x15 is the minimum
//
// Region sizes should be determined by the size of the face, not by the size of the image

void find_smile_using_haar_cascades (FaceAndEyes& this_face)
{
	const float ideal_smile_width  = this_face.face.width * tcG;
	const float ideal_smile_height = this_face.face.height * tcH;
	const int min_threshold_W = ideal_smile_width/1.6;
	const int MaxThresholdW = ideal_smile_width*1.6;
	const int MinThresholdH = ideal_smile_height/1.6;
	const int MaxThresholdH = ideal_smile_height*1.6;

	// greyscale_copy is still where it was, and does not need to be regenerated

	this_face.smile.list.clear();
	this_face.smile.lis2.clear();

	this_face.smile.ROIRect.x = lx(this_face.face);
	this_face.smile.ROIRect.y = my(this_face.face);
	this_face.smile.ROIRect.width = this_face.face.width;
	this_face.smile.ROIRect.height = this_face.face.height/2;

	// a common-sense check: smiles cannot be more than three-quarters of the width of the face
	// removed as no longer necessary
	// if (MaxThreshold > (this_face.smile.ROIRect.width*3/4))  MaxThreshold = this_face.smile.ROIRect.width*3/4;

	run_cascade_and_store(smile_cascade,this_face.smile.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'s',this_face.smile.list);

	run_cascade_and_store(mouth_cascade,this_face.smile.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'m',this_face.smile.lis2);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// A typical nose region in a 250x250 image might be 30x30.
// Anything much larger than that isn't a nose, and anything much smaller than that is too small to be useful.
// Set up the thresholds so that 50x50 is the maximum and 15x15 is the minimum

void find_nose_using_haar_cascades (FaceAndEyes& this_face)
{
	// greyscale_copy is still where it was, and does not need to be regenerated

	const float ideal_nose_width  = this_face.face.width * tcJ;
	const float ideal_nose_height = this_face.face.height * tcK;
	const int min_threshold_W = ideal_nose_width/1.6;
	const int MaxThresholdW = ideal_nose_width*1.6;
	const int MinThresholdH = ideal_nose_height/1.6;
	const int MaxThresholdH = ideal_nose_height*1.6;

	this_face.nose.list.clear();

	this_face.nose.ROIRect = this_face.face;

	// a common-sense check: noses cannot be more than half the width of the face
	// no longer required
	// if (MaxThreshold > mx(this_face.nose.ROIRect))  MaxThreshold = mx(this_face.nose.ROIRect);

	run_cascade_and_store(nose_cascade,this_face.nose.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'n',this_face.nose.list);

	if (qWhichNoseCascade == 2)
	{
		run_cascade_and_store(nos2_cascade,this_face.nose.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'p',this_face.nose.lis2);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Note that this follows on from finding faces, and uses the face vector set up previously
// A typical eye region in a 250x250 image might be 30x18.
// Anything much larger than that isn't an eye, and anything much smaller than that is too small to be useful.
// Set up the thresholds so that 50x50 is the maximum and 12x12 is the minimum
// Although most gaze prediction methods use 40x15 eye regions, OpenCV's eye detectors return square regions

void find_eyes_using_haar_cascades (FaceAndEyes& this_face)
{
	if (eye_cascade_count == 1)  return;  // not implemented yet

	const float ideal_eye_width  = this_face.face.width * tcM;
	const float ideal_eye_height = this_face.face.height * tcN;
	const int min_threshold_W = ideal_eye_width/1.6;
	const int MaxThresholdW = ideal_eye_width*1.6;
	const int MinThresholdH = ideal_eye_height/1.6;
	const int MaxThresholdH = ideal_eye_height*1.6;

	// greyscale_copy is still where it was, and does not need to be regenerated

	this_face.leye.list.clear();
	this_face.reye.list.clear();

	this_face.leye.ROIRect.x = this_face.face.x + (this_face.face.width/3);  // left eyes are on the right as seen on the screen
	this_face.reye.ROIRect.x = this_face.face.x;
	this_face.leye.ROIRect.y = this_face.reye.ROIRect.y = this_face.face.y;
	this_face.leye.ROIRect.width = this_face.reye.ROIRect.width = this_face.face.width*2/3;
	this_face.leye.ROIRect.height = this_face.reye.ROIRect.height = (this_face.smile.lis2.size() == 1)
																	? this_face.smile.lis2[0].box.y-this_face.leye.ROIRect.y
																	: (this_face.smile.list.size() == 1)
																	? this_face.smile.list[0].box.y-this_face.leye.ROIRect.y
																	: this_face.face.height/2;

	// a common-sense check: eyes cannot be more than half the width of the face
	// no longer necessary
	// if (MaxThreshold > mx(this_face.leye.ROIRect))  MaxThreshold = mx(this_face.leye.ROIRect);

	run_cascade_and_store(eye1_cascade,this_face.leye.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'l',this_face.leye.list);
	run_cascade_and_store(eye2_cascade,this_face.reye.ROIRect,cv::Size(min_threshold_W,MinThresholdH),cv::Size(MaxThresholdW,MaxThresholdH),'r',this_face.reye.list);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool load_haar_cascades (void)
{
	const char *face_cascade_name = (qWhichFaceCascade == 2) ? flocHaarFace2 : (qWhichFaceCascade == 1) ? flocHaarFace1 : flocHaarFace0;
	const char *fac2_cascade_name = (qWhichFaceCascade == 0) ? flocHaarFace2 : flocHaarFace0;

    if (!face_cascade.load(face_cascade_name))
    {
		std::clog <<  "Unable to load cascade " << face_cascade_name << std::endl;
		return false;
    }

    if (!fac2_cascade.load(fac2_cascade_name))
    {
		std::clog <<  "Unable to load cascade " << fac2_cascade_name << std::endl;
		return false;
    }

    if (!smile_cascade.load(flocHaarSmile))
	{
		std::clog <<  "Unable to load cascade " << flocHaarSmile << std::endl;
		return false;
	}

    if (!mouth_cascade.load(flocHaarMouth))
	{
		std::clog <<  "Unable to load cascade " << flocHaarMouth << std::endl;
		return false;
	}

	if (qWhichNoseCascade == NOSE_MCS)
	{
	    if (!nose_cascade.load(flocHaarNose))
		{
			std::clog <<  "Unable to load cascade " << flocHaarNose << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == NOSE_FRONTAL_ONLY)
	{
	    if (!nose_cascade.load(flocHaarFNose))
		{
			std::clog <<  "Unable to load cascade " << flocHaarFNose << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == NOSE_FRONTAL_19x13)
	{
	    if (!nose_cascade.load(flocHaarFNose19))
		{
			std::clog <<  "Unable to load cascade " << flocHaarFNose19 << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == NOSE_FRONTAL_25x17)
	{
	    if (!nose_cascade.load(flocHaarFNose25))
		{
			std::clog <<  "Unable to load cascade " << flocHaarFNose25 << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == NOSE_FRONTAL_31x21)
	{
	    if (!nose_cascade.load(flocHaarFNose31))
		{
			std::clog <<  "Unable to load cascade " << flocHaarFNose31 << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == NOSE_PROFILE_ONLY)
	{
	    if (!nose_cascade.load(flocHaarPNose))
		{
			std::clog <<  "Unable to load cascade " << flocHaarPNose << std::endl;
			return false;
		}
	}
	else
	if (qWhichNoseCascade == 2)
	{
	    if (!nose_cascade.load(flocHaarFNose))
	    {
			std::clog <<  "Unable to load cascade " << flocHaarFNose << std::endl;
			return false;
	    }

	    if (!nos2_cascade.load(flocHaarPNose))
	    {
			std::clog <<  "Unable to load cascade " << flocHaarPNose << std::endl;
			return false;
	    }
	}

	if (qWhichEyeCascade == 1)
	{
	    if (!eye1_cascade.load(flocHaarLEye))
	    {
			std::clog <<  "Unable to load cascade " << flocHaarEyes << std::endl;
			return false;
	    }
	}
	else
	if (qWhichEyeCascade == 2)
	{
	    if (!eye1_cascade.load(flocHaarLEye))
	    {
			std::clog <<  "Unable to load cascade " << flocHaarLEye << std::endl;
			return false;
	    }

	    if (!eye2_cascade.load(flocHaarREye))
	    {
			std::clog <<  "Unable to load cascade " << flocHaarREye << std::endl;
			return false;
	    }
	}

	qHaveCascades = true;
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

