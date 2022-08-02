#ifndef _LANDMARKS_HPP_
#define _LANDMARKS_HPP_

#include <opencv2/core/core.hpp>

#include "Types.hpp"

////////////////////////////////////////////////////////////////////////////////////////////

void set_smile_landmark (FaceAndEyes& this_face);
void set_eye_landmarks (FaceAndEyes& this_face);
void set_nose_landmark (FaceAndEyes& this_face);

////////////////////////////////////////////////////////////////////////////////////////////

extern bool qUseTidy;

extern FaceAndEyes faces_and_eyes_list[100];
extern int faces_and_eyes_count;

extern void prune_face_list(void);
extern void tidy_eyes_using_common_sense (FaceAndEyes& this_face);
extern void tidy_noses_using_common_sense (FaceAndEyes& this_face);

////////////////////////////////////////////////////////////////////////////////////////////

extern bool qUseKnownFaces;

extern bool use_known_faces_if_available (FaceAndEyes& this_face);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

extern void log_prediction_results (const char *preamble, const cv::Point2i& known, const cv::Point2i& predicted);
extern void log_prediction_results (const char *preamble, const cv::Point2i& known, const cv::Point2f& predicted);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
