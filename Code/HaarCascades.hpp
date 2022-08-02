#ifndef _HAAR_CASCADES_HPP_
#define _HAAR_CASCADES_HPP_

#include "Landmarks.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////

extern bool qUseCascades;
extern bool qHaveCascades;

///////////////////////////////////////////////////////////////////////////////////////////////

extern int qWhichFaceCascade;
extern int qWhichEyeCascade;
extern int qWhichNoseCascade;
#define NOSE_MCS 1
#define NOSE_FRONTAL_ONLY   6
#define NOSE_FRONTAL_19x13 19
#define NOSE_FRONTAL_25x17 25
#define NOSE_FRONTAL_31x21 31
#define NOSE_PROFILE_ONLY  16

///////////////////////////////////////////////////////////////////////////////////////////////

bool load_haar_cascades (void);

///////////////////////////////////////////////////////////////////////////////////////////////

void find_face_using_haar_cascades (void);

void find_smile_using_haar_cascades (FaceAndEyes& this_face);

void find_nose_using_haar_cascades (FaceAndEyes& this_face);

void find_eyes_using_haar_cascades (FaceAndEyes& this_face);

///////////////////////////////////////////////////////////////////////////////////////////////

#endif
