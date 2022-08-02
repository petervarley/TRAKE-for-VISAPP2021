
#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include "Known.hpp"
#include "Types.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool qUseKnown = false;
bool qHaveKnown = false;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

char *known_facename;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool IsKnownPixel (std::map<const std::string,cv::Point2i>& list, cv::Point2i &found)
{
	std::map<const std::string,cv::Point2i>::iterator it = list.find(known_facename);
	if (it == list.end())  return false;
	found = list.find(known_facename)->second;
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownPixels (const char *filename, std::map<const std::string,cv::Point2i>& list)
{
	FILE *KnownEyesList = fopen(filename,"r");

	for (;;)
	{
		char buffer[128];
		fgets(buffer,128,KnownEyesList);
		if (feof(KnownEyesList))  break;

		char name[128];
		cv::Point2i eye;
		if (sscanf(buffer,"%s %d %d",name,&eye.x,&eye.y) == 3)
		{
			//clog2i(name,eye);
			list[name] = eye;
		}
	}

	fclose(KnownEyesList);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static bool is_known_rect (std::map<const std::string,cv::Rect>& list, cv::Rect &found)
{
	std::map<const std::string,cv::Rect>::iterator it = list.find(known_facename);
	if (it == list.end())  return false;
	found = list.find(known_facename)->second;
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void initialise_known_rects (const char *filename, std::map<const std::string,cv::Rect>& list)
{
	FILE *KnownEyesList = fopen(filename,"r");

	for (;;)
	{
		char buffer[128];
		fgets(buffer,128,KnownEyesList);
		if (feof(KnownEyesList))  break;

		char name[128];
		cv::Rect eye;
		if (sscanf(buffer,"%s %d %d %d %d",name,&eye.x,&eye.y,&eye.width,&eye.height) == 5)
		{
			//clogrect(name,eye);
			list[name] = eye;
		}
	}

	fclose(KnownEyesList);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::map<const std::string,cv::Rect> facemap;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_face (cv::Rect &face)
{
	return is_known_rect(facemap,face);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownFaces (void)
{
	initialise_known_rects("KnownFaces.txt",facemap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::map<const std::string,cv::Point2i> lpupilmap;
static std::map<const std::string,cv::Point2i> rpupilmap;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_Lpupil (cv::Point2i &eye)
{
	return IsKnownPixel(lpupilmap,eye);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownLPupils (void)
{
	InitialiseKnownPixels("KnownLPupils.txt",lpupilmap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_Rpupil (cv::Point2i &eye)
{
	return IsKnownPixel(rpupilmap,eye);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownRPupils (void)
{
	InitialiseKnownPixels("KnownRPupils.txt",rpupilmap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::map<const std::string,cv::Rect> leyemap;
static std::map<const std::string,cv::Rect> reyemap;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_Leye (cv::Rect &eye)
{
	return is_known_rect(leyemap,eye);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownLEyes (void)
{
	initialise_known_rects("KnownLEyes.txt",leyemap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_Reye (cv::Rect &eye)
{
	return is_known_rect(reyemap,eye);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownREyes (void)
{
	initialise_known_rects("KnownREyes.txt",reyemap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::map<const std::string,cv::Point2i> nosemap;
static std::map<const std::string,cv::Rect> mouthmap;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_nose (cv::Point2i &nose)
{
	return IsKnownPixel(nosemap,nose);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownNoses (void)
{
	InitialiseKnownPixels("KnownNoses.txt",nosemap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_known_mouth (cv::Rect &mouth)
{
	return is_known_rect(mouthmap,mouth);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void InitialiseKnownMouths (void)
{
	initialise_known_rects("KnownMouths.txt",mouthmap);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void initialise_known (void)
{
	if (!qHaveKnown)
	{
		InitialiseKnownFaces();
		InitialiseKnownLPupils();
		InitialiseKnownRPupils();
		InitialiseKnownLEyes();
		InitialiseKnownREyes();
		InitialiseKnownNoses();
		InitialiseKnownMouths();
		qHaveKnown = true;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

bool load_known_frame (const char *Facename)
{
	known_facename = Facename;
	//std::clog << "Gaze Prediction from known face " << known_facename << std::endl;
	FaceAndEyes this_frame;

	cv::Rect face;
	if (is_known_face(face))  this_frame.face = face;
	else return false;

	cv::Rect mouth;
	if (is_known_mouth(mouth))  this_frame.smile.best.box = mouth;
	else return false;

	cv::Rect eye;
	if (is_known_Leye(eye))  this_frame.leye.best.box = eye;	else return false;
	if (is_known_Reye(eye))  this_frame.reye.best.box = eye;	else return false;

	cv::Rect nose;
	cv::Point2i tip;
	if (is_known_nose(tip))
	{
		nose.x = tip.x-3;
		nose.y = tip.y-3;
		nose.width = nose.height = 7;
		this_frame.nose.best.box = nose;
	}
	else return false;

	cv::Point2i pupil;
	if (is_known_Lpupil(pupil))  this_frame.lpupil = pupil;	else return false;
	if (is_known_Rpupil(pupil))  this_frame.rpupil = pupil;	else return false;

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
