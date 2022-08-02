
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "CVFrame.hpp"
#include "HaarCascades.hpp"
#include "Landmarks.hpp"
#include "EyeRegions.hpp"
#include "Pupils.hpp"
#include "Noses.hpp"
#include "Mouths.hpp"
#include "Constants.hpp"
#include "Known.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat working_frame;
cv::Mat greyscale_copy;
cv::Mat shrunk_frame;

void create_greyscale_copy (void)
{
    cvtColor(working_frame,greyscale_copy,CV_BGR2GRAY);
    equalizeHist(greyscale_copy,greyscale_copy);
}

void create_shrunk_greyscale_copy (void)
{
    cvtColor(shrunk_frame,greyscale_copy,CV_BGR2GRAY);
    equalizeHist(greyscale_copy,greyscale_copy);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void blur_and_equalise (const cv::Mat& MatIn, cv::Mat& MatOut)
{
    cv::blur(MatIn,MatOut,cv::Size(3,3));
    cv::equalizeHist(MatOut,MatOut);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_frame_to_png_file (const char *filename, cv::Mat& frame)
{
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

	try
	{
	    cv::imwrite(filename, frame, compression_params);
		std::clog <<  "Written frame as PNG to image file " << filename << std::endl;
	}
	catch (const std::exception& e)
	{
		std::clog <<  "Couldn't write frame as PNG to image file " << filename << e.what() << std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void write_frame_to_jpg_file (const char *filename, cv::Mat& frame)
{
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(27);

	try
	{
    	cv::imwrite(filename, frame, compression_params);
		std::clog <<  "Written frame as JPG to image file " << filename << std::endl;
	}
	catch (const std::exception& e)
	{
		std::clog <<  "Couldn't write frame as JPG to image file " << filename << e.what()  << std::endl;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void write_frame_to_file (const char *filename, cv::Mat& frame)
{
	if (strstr(filename,".jpg") || strstr(filename,".JPG"))
	{
		write_frame_to_jpg_file(filename,frame);
	}
	else
	{
		write_frame_to_png_file(filename,frame);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

static cv::Scalar FoundFaceColour (160, 160, 255);
static cv::Scalar FoundSmileColour (232, 108, 200);
static cv::Scalar FoundNoseColour (255, 192, 180);
static cv::Scalar FoundLEyeColour (28, 255, 164);
static cv::Scalar FoundREyeColour (52, 255, 228);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

int qFrameSpeed = 100;
bool qShowFaceWindow = true;
bool qSaveFrames = false;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool read_working_frame (const char *from_directory, const char *filename, const char *options)
{
	std::clog << "Read Working Frame " << from_directory << "/" << filename << std::endl;
	known_facename = filename;

	char input_filename[256];
	sprintf(input_filename,"%s/%s",from_directory,filename);

    working_frame = cv::imread(input_filename,CV_LOAD_IMAGE_COLOR);
	// std::clog <<  "Read frame from file " << input_filename << std::endl;

    if (working_frame.data == NULL)
    {
		std::clog <<  "Error reading file " << input_filename << ", no data" << std::endl;
		return false;
    }

	create_greyscale_copy();
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void show_frame_on_window (int maxw, int maxh)
{
	int w = working_frame.cols;
	int h = working_frame.rows;

	if ((w > maxw) || (h > maxh))
	{
		float wr = (float)w/(float)maxw;
		float hr = (float)h/(float)maxh;
		float mr = (wr > hr) ? wr : hr;
		w /= mr;
		h /= mr;
	}

   	cv::resizeWindow(cSourceWindowName,w,h);
	cv::imshow(cSourceWindowName, working_frame);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void many_rectangles (cv::Mat& working_frame, std::vector<Region>& list, const cv::Scalar& colour, int width)
{
	for (int i=0; i<list.size(); ++i)  rectangle(working_frame,list[i].box,colour,2);
}

void read_and_display_frame (const char *from_directory, const char *filename, const char *to_directory, const char *options)
{
	qSaveFrames = (strchr(options,'w') != nullptr);

	if (read_working_frame(from_directory,filename,options))
    {
		if (qUseKnownFaces && use_known_faces_if_available(faces_and_eyes_list[0]))
		{
			faces_and_eyes_count = 1;
		}
		else
		if (qUseCascades)  find_face_using_haar_cascades();

		if (!faces_and_eyes_count)  // hack something to keep the process going
		{
			faces_and_eyes_count = 1;
			faces_and_eyes_list[0].face.x = 6;
			faces_and_eyes_list[0].face.y = 6;
			faces_and_eyes_list[0].face.width = working_frame.cols-12;
			faces_and_eyes_list[0].face.height = working_frame.rows-12;
		}

		for (int i=0; i<faces_and_eyes_count; ++i)
		{
			faces_and_eyes_list[i].distance = tcZ*tcW/(float)faces_and_eyes_list[i].face.width;
			estimate_initial_mouths(faces_and_eyes_list[i]);
			estimate_initial_eye_regions(faces_and_eyes_list[i]);
			estimate_initial_noses(faces_and_eyes_list[i]);
		}

		prune_face_list();

		for (int i=0; i<faces_and_eyes_count; ++i)  if (faces_and_eyes_list[i].process_this)
		{
			estimate_initial_pupils(faces_and_eyes_list[i]);
		    estimate_improved_eye_regions(faces_and_eyes_list[i]);
		    estimate_improved_noses(faces_and_eyes_list[i]);

		    if (qUseKnownMouths)  use_known_mouths_if_available(faces_and_eyes_list[i]);

		    if (qUseKnownEyes)  use_known_eyes_if_available(faces_and_eyes_list[i]);

		    locate_accurate_noses(faces_and_eyes_list[i]);
		    if (qUseKnownNoses)  use_known_nose_if_available(faces_and_eyes_list[i]);

			if (qShowFaceWindow)
			{
				rectangle(working_frame,faces_and_eyes_list[i].face,FoundFaceColour,2);
				rectangle(working_frame,faces_and_eyes_list[i].smile.best.box,FoundSmileColour,2);
				rectangle(working_frame,faces_and_eyes_list[i].leye.best.box,FoundLEyeColour,2);
				rectangle(working_frame,faces_and_eyes_list[i].reye.best.box,FoundREyeColour,2);

				if (qWhichNoseCascade != 2)
				{
					rectangle(working_frame,faces_and_eyes_list[i].nose.best.box,FoundNoseColour,2);
				}
				else
				{
					many_rectangles(working_frame,faces_and_eyes_list[i].nose.list,FoundNoseColour,2);
					many_rectangles(working_frame,faces_and_eyes_list[i].nose.lis2,FoundNoseColour,2);
				}
			}
		}

		if (qShowFaceWindow)  show_frame_on_window(800,400);

		if (qSaveFrames)
		{
			char output_filename[256];
			sprintf(output_filename,"%s/%s",to_directory,filename);
			write_frame_to_file(output_filename,working_frame);
		}
    }

    cv::waitKey(qFrameSpeed);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void setup_source_window (void)
{
	static bool SourceWindowExists = false;

	if (!SourceWindowExists)
	{
		std::clog <<  "Trying to create face window " << cSourceWindowName << std::endl;

		if (qShowFaceWindow)
		{
	    	cv::namedWindow(cSourceWindowName, cv::WINDOW_NORMAL );// Create a window for echoing camera input
	    	cv::resizeWindow(cSourceWindowName,250,250);
	    	cv::moveWindow(cSourceWindowName,100,100);
	    	SourceWindowExists = true;
		}

		if (qShowEyeWindows)
		{
    		cv::namedWindow(cLEyeWindowName, cv::WINDOW_NORMAL );
    		cv::namedWindow(cREyeWindowName, cv::WINDOW_NORMAL );
		}

		if (qShowNoseWindow)
		{
    		cv::namedWindow(cNoseWindowName, cv::WINDOW_NORMAL );
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
