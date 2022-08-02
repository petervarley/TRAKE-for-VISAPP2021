#ifndef _KNOWN_HPP_
#define _KNOWN_HPP_

#include <opencv2/core/core.hpp>

extern bool qUseKnown;
extern bool qHaveKnown;

void initialise_known (void);

extern char *known_facename;

bool is_known_face (cv::Rect &face);

bool is_known_Lpupil (cv::Point2i &pupil);

bool is_known_Rpupil (cv::Point2i &pupil);

bool is_known_Leye (cv::Rect &eye);

bool is_known_Reye (cv::Rect &eye);

bool is_known_nose (cv::Point2i &nose);

bool is_known_mouth (cv::Rect &mouth);

bool load_known_frame (const char *Facename);

#endif
