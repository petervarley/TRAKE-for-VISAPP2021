#ifndef _CV_FRAME_HPP_
#define _CV_FRAME_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////////////

extern int qFrameSpeed;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void write_frame_to_png_file (const char *filename, cv::Mat& frame);
void write_frame_to_jpg_file (const char *filename, cv::Mat& frame);
void write_frame_to_file (const char *filename, cv::Mat& frame);

void read_and_display_frame (const char *from_directory, const char *filename, const char *to_directory, const char *options);
bool read_working_frame (const char *from_directory, const char *filename, const char *options);

void setup_source_window (void);

extern bool qShowFaceWindow;
extern bool qShowEyeWindows;
extern bool qSaveFrames;

/////////////////////////////////////////////////////////////////////////////////////////////////////////

// The Working frame is the input RGB
// it is not used by image processing routines, but it is written to by markup
extern cv::Mat working_frame;

// The Greyscale copy of the working frame is the one used by most image processing routines
extern cv::Mat greyscale_copy;

// The Shrunk frame is a copy of the working frame used for processing oversized images
extern cv::Mat shrunk_frame;

// A standard conversion routine to make sure training and use do things the same way
void create_greyscale_copy (void);
void create_shrunk_greyscale_copy (void);

// What it says

void blur_and_equalise (const cv::Mat& MatIn, cv::Mat& MatOut);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
