
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "Landmarks.hpp"
#include "Pupils.hpp"
#include "Known.hpp"
#include "Rect.hpp"
#include "Stats.hpp"
#include "Constants.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

static void GetVectorBoundingBox (cv::Rect& BBox, const std::vector<Region>& list)
{
	int L = list[0].box.x;
	int R = L+list[0].box.width;
	int T = list[0].box.y;
	int B = T+list[0].box.height;

	for (int N=1;  N<list.size();  ++N)
	{
		if (list[N].box.x < L)  L = list[N].box.x;
		if (list[N].box.y < T)  T = list[N].box.y;

		int RN = list[N].box.x + list[N].box.width;
		int BN = list[N].box.y + list[N].box.height;

		if (RN > R)  R = RN;
		if (BN > B)  B = BN;
	}

	BBox.x = L;
	BBox.y = T;
	BBox.width = R-L;
	BBox.height = B-T;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_smile_landmark (FaceAndEyes& this_face)
{
	if ((this_face.smile.list.size() == 1) && (this_face.smile.lis2.size() == 1))
	{
		// spoilt for choice, use the average
		this_face.smile.best.box.x = (this_face.smile.list[0].box.x+this_face.smile.lis2[0].box.x)/2;
		this_face.smile.best.box.y = (this_face.smile.list[0].box.y+this_face.smile.lis2[0].box.y)/2;
		this_face.smile.best.box.width = (this_face.smile.list[0].box.width+this_face.smile.lis2[0].box.width)/2;
		this_face.smile.best.box.height = (this_face.smile.list[0].box.height+this_face.smile.lis2[0].box.height)/2;
	}
	else if (this_face.smile.list.size() == 1)
	{
		this_face.smile.best = this_face.smile.list[0];
	}
	else if (this_face.smile.lis2.size() == 1)
	{
		this_face.smile.best = this_face.smile.lis2[0];
	}
	else if ((this_face.smile.list.size() > 1) && (this_face.smile.lis2.size() == 0))
	{
		// bounding box of all the detected smiles
		// std::clog << std::to_string(this_facesmile_list.size()) << "smiles detected, bounding box" << std::endl;
		GetVectorBoundingBox(this_face.smile.best.box,this_face.smile.list);
	}
	else if ((this_face.smile.lis2.size() > 1) && (this_face.smile.list.size() == 0))
	{
		// bounding box of all the detected mouths
		// std::clog << std::to_string(this_facesmile_list.size()) << "smiles detected, bounding box" << std::endl;
		GetVectorBoundingBox(this_face.smile.best.box,this_face.smile.lis2);
	}
	else // both zero
	{
		// the best we can do is make a wild guess

		//The OpenCV smile detector typically returns a rectangle which is 44% of the width and 22% of the height of the face.
		// The mid-point is typically 78% down from the top of the face.

		// The mcs mouth detector typically returns a rectangle which is 38% of the width and 23% of the height of the face.
		// The mid-point is typically 80% down from the top of the face.

		int midx = mx(this_face.face);
		int midy = this_face.face.y + (int)(this_face.face.height*tcF);
		int widx = (int)(this_face.face.width*tcG);
		int heiy = (int)(this_face.face.height*tcH);

		this_face.smile.best.box.x = midx-widx/2;
		this_face.smile.best.box.y = midy-heiy/2;
		this_face.smile.best.box.width = widx;
		this_face.smile.best.box.height = heiy;
		this_face.smile.best.tag = '?';
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_eye_landmarks (FaceAndEyes& this_face)
{
	if (this_face.leye.list.size() == 0)
	{
		// the best we can do is make a wild guess
		int midy = this_face.face.y + (int)(this_face.face.height*tcL);
		int widx = (int)(this_face.face.width*tcM);
		int heiy = (int)(this_face.face.height*tcN);
		int ofsx = (int)(this_face.face.width*tcO/2.0);
		int midx = mx(this_face.face)+ofsx;

		this_face.leye.best.box.x = midx-widx/2;
		this_face.leye.best.box.y = midy-heiy/2;
		this_face.leye.best.box.width = widx;
		this_face.leye.best.box.height = heiy;
		this_face.leye.best.tag = '?';
	}
	else if (this_face.leye.list.size() == 1)
	{
		this_face.leye.best = this_face.leye.list[0];
	}
	else
	{
		// bounding box of all the detected faces
		GetVectorBoundingBox(this_face.leye.best.box,this_face.leye.list);
	}

	if (this_face.reye.list.size() == 0)
	{
		// the best we can do is make a wild guess
		int midy = this_face.face.y + (int)(this_face.face.height*tcL);
		int widx = (int)(this_face.face.width*tcM);
		int heiy = (int)(this_face.face.height*tcN);
		int ofsx = (int)(this_face.face.width*tcO/2.0);
		int midx = mx(this_face.face)-ofsx;

		this_face.reye.best.box.x = midx-widx/2;
		this_face.reye.best.box.y = midy-heiy/2;
		this_face.reye.best.box.width = widx;
		this_face.reye.best.box.height = heiy;
		this_face.reye.best.tag = '?';
	}
	else if (this_face.reye.list.size() == 1)
	{
		this_face.reye.best = this_face.reye.list[0];
	}
	else
	{
		// bounding box of all the detected faces
		GetVectorBoundingBox(this_face.reye.best.box,this_face.reye.list);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void set_nose_landmark (FaceAndEyes& this_face)
{
	if (this_face.nose.list.size() == 0)
	{
		// the best we can do is make a wild guess
		// The mcs nose detector typically returns a rectangle which is 30% of the width and 25% of the height of the face.
		// The mid-point is typically 60% down from the top of the face.
		//
		// Redoing this with known faces and noses, the median vertical displacement is nearer 58.3%.
		// although the face cascades are so inaccurate that it could reasonably be anywhere from 49% to 68%.

		int midx = mx(this_face.face);
		int midy = this_face.face.y + (int)(this_face.face.height*tcI);
		int widx = (int)(this_face.face.width*tcJ);
		int heiy = (int)(this_face.face.height*tcK);

		this_face.nose.best.box.x = midx-widx/2;
		this_face.nose.best.box.y = midy-heiy/2;
		this_face.nose.best.box.width = widx;
		this_face.nose.best.box.height = heiy;
		this_face.nose.best.tag = '?';
	}
	else if (this_face.nose.list.size() == 1)
	{
		this_face.nose.best = this_face.nose.list[0];
	}
	else
	{
		float ideal_nose_height = ((float)this_face.face.height) * tcK;
		float ideal_nose_midy = this_face.face.y + (((float)this_face.face.height) * tcI) - (ideal_nose_height/2.0);

		float ideal_nose_width = ((float)this_face.face.width) * tcJ;

		float ideal_nose_y = this_face.face.y + (((float)this_face.face.height) * tcI) - (ideal_nose_height/2.0);
		float ideal_nose_x = this_face.face.x + ((this_face.face.width-ideal_nose_width)/2.0);
		cv::Rect ideal;
		ideal.x = ideal_nose_x;
		ideal.y = ideal_nose_y;
		ideal.width = ideal_nose_width;
		ideal.height = ideal_nose_height;
		//clogrect("... ideal",ideal);

		int closest_so_far = -1;
		float closest_distance;

		for (int N=0; N<this_face.nose.list.size(); ++N)
		{
			float actual_nose_height = this_face.nose.list[N].box.height;
			float actual_nose_midy = this_face.nose.list[N].box.y + this_face.nose.list[N].box.height/2.0;

			float dh = actual_nose_height-ideal_nose_height;
			float dy = actual_nose_midy-ideal_nose_midy;

			float distance = dh*dh + dy*dy;

			if ((closest_so_far < 0) || (distance < closest_distance))
			{
				closest_so_far = N;
				closest_distance = distance;
			}
		}

		this_face.nose.best = this_face.nose.list[closest_so_far];
	}
	clogregion("Best Nose",this_face.nose.best);
}

/////////////////////////////////////////////////////////////////////////////////////

bool qUseTidy = true;

///////////////////////////////////////////////////////////////////////////////////////////////////////

FaceAndEyes faces_and_eyes_list[100];
int faces_and_eyes_count = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------------------------------

static bool IsLeftEye (cv::Rect& P, cv::Rect& Q, cv::Rect& X)  // remember that left eyes are on the right and vice versa
{
	return ((mx(P) + mx(Q))/2) > mx(X);
}

//-----------------------------------------------------------------------------------------------------
// 1. Resize eye regions so that they are in proportion to the face region (26% seems typical)

static void ResizeEyesInProportionToFace (std::vector<Region>& eye_list, int face_size)
{
	float ideal_eye_size = ((float)face_size) * 0.26;  // another arbitrary constant

	for (int L = 0;  L < eye_list.size();  ++L)
	{
		float current_eye_size = eye_list[L].box.width;
		float new_eye_size = ((ideal_eye_size*1.6) + current_eye_size)/2.6;

		float CX = mx(eye_list[L].box);
		float CY = my(eye_list[L].box);

		float XL = CX - new_eye_size/2.0;
		float XR = CX + new_eye_size/2.0;
		float YT = CY - new_eye_size/2.0;
		float YB = CY + new_eye_size/2.0;

		eye_list[L].box.x = (int)XL;
		eye_list[L].box.y = (int)YT;
		eye_list[L].box.width = (int)(XR-XL);
		eye_list[L].box.height = (int)(YB-YT);
	}
}

//-----------------------------------------------------------------------------------------------------
// 2. If a face has exactly one left eye, remove any right eyes which overlap it

static void remove_wrong_eyes_which_overlap (FaceAndEyes& this_face_and_eyes)
{
	std::vector<bool>eyesl_valid;
	std::vector<bool>eyesr_valid;

	for (int L = 0;  L < this_face_and_eyes.leye.list.size();  ++L)  eyesl_valid.push_back(true);
	for (int R = 0;  R < this_face_and_eyes.reye.list.size();  ++R)  eyesr_valid.push_back(true);

	for (int L = 0;  L < this_face_and_eyes.leye.list.size();  ++L)
	for (int R = 0;  R < this_face_and_eyes.reye.list.size();  ++R)
	{
		if (o2(this_face_and_eyes.leye.list[L].box,this_face_and_eyes.reye.list[R].box))
		{
			if (IsLeftEye(this_face_and_eyes.leye.list[L].box,this_face_and_eyes.reye.list[R].box,this_face_and_eyes.face))
			{
				// invalidate the right eye rect
				eyesr_valid[R] = false;
			}
			else
			{
				// invalidate the left eye rect
				eyesl_valid[L] = false;
			}
		}
	}

	std::vector<Region>new_eyesl;
	std::vector<Region>new_eyesr;

	for (int L = 0;  L < this_face_and_eyes.leye.list.size();  ++L)  if (eyesl_valid[L])  new_eyesl.push_back(this_face_and_eyes.leye.list[L]);
	this_face_and_eyes.leye.list = new_eyesl;

	for (int R = 0;  R < this_face_and_eyes.reye.list.size();  ++R)  if (eyesr_valid[R])  new_eyesr.push_back(this_face_and_eyes.reye.list[R]);
	this_face_and_eyes.reye.list = new_eyesr;
}

//-----------------------------------------------------------------------------------------------------

static Region reflect_eye_across_face (Region& OneEye, cv::Rect& Face)
{
	Region other_eye;
	other_eye.box.y = OneEye.box.y;
	other_eye.box.height = OneEye.box.height;
	other_eye.box.x = Face.x + Face.x + Face.width - OneEye.box.x - OneEye.box.width;
	other_eye.box.width = OneEye.box.width;
	other_eye.tag = OneEye.tag;
	return other_eye;
}

//-----------------------------------------------------------------------------------------------------
// 5. If there is one left eye and more than one right eye (or vice versa), pick the one nearest where the reflection says it should be

static float distance_between_eye_centres (cv::Rect& L, cv::Rect& R)
{
	float LX = L.x + ((float)L.width)/2.0;
	float LY = L.y + ((float)L.height)/2.0;

	float RX = R.x + ((float)R.width)/2.0;
	float RY = R.y + ((float)R.height)/2.0;

	float DX = LX-RX;
	float DY = LY-RY;
	return (DX*DX)+(DY*DY);
}

static void PickBestReflections (FaceAndEyes& this_face_and_eyes)
{
	if ((this_face_and_eyes.leye.list.size() == 1) && (this_face_and_eyes.reye.list.size() > 1))
	{
		Region TargetEye = reflect_eye_across_face(this_face_and_eyes.leye.list[0],this_face_and_eyes.face);

		Region best_choice_eye = this_face_and_eyes.reye.list[0];
		float BestDistance = distance_between_eye_centres(TargetEye.box,best_choice_eye.box);

		for (int R = 1;  R < this_face_and_eyes.reye.list.size();  ++R)
		{
			float NewDistance = distance_between_eye_centres(TargetEye.box,this_face_and_eyes.reye.list[R].box);

			if (NewDistance < BestDistance)
			{
				BestDistance = NewDistance;
				best_choice_eye = this_face_and_eyes.reye.list[R];
			}
		}

		this_face_and_eyes.reye.list.clear();
		this_face_and_eyes.reye.list.push_back(best_choice_eye);
	}
	else
	if ((this_face_and_eyes.leye.list.size() > 1) && (this_face_and_eyes.reye.list.size() == 1))
	{
		Region TargetEye = reflect_eye_across_face(this_face_and_eyes.reye.list[0],this_face_and_eyes.face);

		Region best_choice_eye = this_face_and_eyes.leye.list[0];
		float BestDistance = distance_between_eye_centres(TargetEye.box,best_choice_eye.box);

		for (int L = 1;  L < this_face_and_eyes.leye.list.size();  ++L)
		{
			float NewDistance = distance_between_eye_centres(TargetEye.box,this_face_and_eyes.leye.list[L].box);

			if (NewDistance < BestDistance)
			{
				BestDistance = NewDistance;
				best_choice_eye = this_face_and_eyes.leye.list[L];
			}
		}

		this_face_and_eyes.leye.list.clear();
		this_face_and_eyes.leye.list.push_back(best_choice_eye);
	}

}

//-----------------------------------------------------------------------------------------------------

static void estimate_missing_eye_by_reflection (FaceAndEyes& this_face_and_eyes)
{
	if ((this_face_and_eyes.leye.list.size() == 1) && (this_face_and_eyes.reye.list.size() == 0))
	{
		Region missing_eye = reflect_eye_across_face(this_face_and_eyes.leye.list[0],this_face_and_eyes.face);
		this_face_and_eyes.reye.list.push_back(missing_eye);
	}
	else
	if ((this_face_and_eyes.leye.list.size() == 0) && (this_face_and_eyes.reye.list.size() == 1))
	{
		Region missing_eye = reflect_eye_across_face(this_face_and_eyes.reye.list[0],this_face_and_eyes.face);
		this_face_and_eyes.leye.list.push_back(missing_eye);
	}
}

//-----------------------------------------------------------------------------------------------------
//2. If the same detector (left or right eye) detects two eyes which overlap, merge them

static cv::Rect merge_eye_boxes (const cv::Rect& P, const cv::Rect& Q)
{
	cv::Rect N;
	N.x = (P.x+Q.x)/2;
	N.y = (P.y+Q.y)/2;
	N.width = (P.width+Q.width)/2;
	N.height = (P.height+Q.height)/2;
	return N;
}

static bool merge_overlapping_eyes (std::vector<Region>& eye_list)
{
	bool MergedSome = false;

	std::vector<bool>eyes_valid;
	for (int L = 0;  L < eye_list.size();  ++L)  eyes_valid.push_back(true);

	std::vector<Region>new_eyes;

	for (int L = 0;  L < eye_list.size(); ++L)  for (int M = L+1;  M < eye_list.size(); ++M)
	if (eyes_valid[L] && eyes_valid[M])
	{
		if (o2(eye_list[L].box,eye_list[M].box))
		{
			Region merged_eye;
			merged_eye.box = merge_eye_boxes(eye_list[L].box,eye_list[M].box);
			merged_eye.tag = 'x';
			new_eyes.push_back(merged_eye);
			eyes_valid[L] = false;
			eyes_valid[M] = false;
			MergedSome = true;
		}
	}

	for (int L = 0;  L < eye_list.size();  ++L)  if (eyes_valid[L])  new_eyes.push_back(eye_list[L]);
	eye_list = new_eyes;

	return MergedSome;
}

//-----------------------------------------------------------------------------------------------------
// 1. Resize nose regions so that they are in proportion to the face region
// again, a width of 26% of the face size seems typical)
// surprisingly, nose reqions are typically wider than tall, and the height is nearer 22% of the face size
//
// This was for use with the mcs nose cascade, and should no longer be required

static void ResizeNosesInProportionToFace (std::vector<cv::Rect>& nose_list, int face_size)
{
	float ideal_nose_width = ((float)face_size) * tcJ;
	float ideal_nose_height = ((float)face_size) * tcK;

	for (int L = 0;  L < nose_list.size();  ++L)
	{
		float current_nose_width = nose_list[L].width;
		float new_nose_width = ((ideal_nose_width*1.6) + current_nose_width)/2.6;

		float current_nose_height = nose_list[L].height;
		float new_nose_height = ((ideal_nose_height*1.6) + current_nose_height)/2.6;

		float CX = mx(nose_list[L]);
		float CY = my(nose_list[L]);

		float XL = CX - new_nose_width/2.0;
		float XR = CX + new_nose_width/2.0;
		float YT = CY - new_nose_height/2.0;
		float YB = CY + new_nose_height/2.0;

		nose_list[L].x = (int)XL;
		nose_list[L].y = (int)YT;
		nose_list[L].width = (int)(XR-XL);
		nose_list[L].height = (int)(YB-YT);
	}
}

//-----------------------------------------------------------------------------------------------------

static void PruneNoseList (FaceAndEyes& this_face)
{
	clogi("... nose count before pruning",this_face.nose.list.size());

	// first rule: the top of the nose must be below the eye midpoint
	// The eye mid-point is typically 35% down from the top of the face.
	if (this_face.nose.list.size() > 1)
	{
		//for (int N=0; N<this_face.nose.list.size(); ++N)  clogregion("... nose(a)",this_face.nose.list[N]);

		std::vector<Region> temp_nose_list;

		int threshold = this_face.face.y + (int)((float)this_face.face.height*0.35);

		temp_nose_list.clear();
		for (int N=0; N<this_face.nose.list.size(); ++N)
		{
			if (this_face.nose.list[N].box.y > threshold)  temp_nose_list.push_back(this_face.nose.list[N]);
		}

		if ((temp_nose_list.size() > 0) && (temp_nose_list.size() < this_face.nose.list.size()))
		{
			this_face.nose.list = temp_nose_list;
		}
	}

	// second rule: the bottom of the nose must be above the mouth midpoint
	// The mouth mid-point is typically 79% down from the top of the face.
	if (this_face.nose.list.size() > 1)
	{
		//for (int N=0; N<this_face.nose.list.size(); ++N)  clogregion("... nose(b)",this_face.nose.list[N]);

		std::vector<Region> temp_nose_list;

		int threshold = this_face.face.y + (int)((float)this_face.face.height*0.79);

		temp_nose_list.clear();
		for (int N=0; N<this_face.nose.list.size(); ++N)
		{
			if ((this_face.nose.list[N].box.y+this_face.nose.list[N].box.height) < threshold)  temp_nose_list.push_back(this_face.nose.list[N]);
		}

		if ((temp_nose_list.size() > 0) && (temp_nose_list.size() < this_face.nose.list.size()))
		{
			this_face.nose.list = temp_nose_list;
		}
	}

	// third rule: the centre of the nose must be between lines from the extremities of the eye and mouth regions
	//
	// this is where it gets tricky
	//
	// mouth detectors typically return a rectangle which is 41% of the width of the face.
	// the mid-point is typically 79% down from the top of the face.
	//
	// left and right eye detectors typically return a square which is 23% of the width of the face.
	// the mid-point is typically 35% down from the top of the face
	// the horizontal distance between the two mid-points is typically 30% of the width of the face.
	//
	// the nose mid-point is typically 60% down from the top of the face.
	//
	// so, to a first approximation ...
	//
	// the left-hand boundary is typically at  8.5% of the width of the face at 35% down from the top of the face (eyes)
	// the left-hand boundary is typically at 20.4% of the width of the face at 60% down from the top of the face (nose)
	// the left-hand boundary is typically at 29.5% of the width of the face at 79% down from the top of the face (mouth)

	if (this_face.nose.list.size() > 1)
	{
		//for (int N=0; N<this_face.nose.list.size(); ++N)  clogregion("... nose(c)",this_face.nose.list[N]);

		std::vector<Region> temp_nose_list;

		cv::Point2f lthresholde(mx(this_face.reye.best.box),my(this_face.reye.best.box));

		cv::Point2f rthresholde(mx(this_face.leye.best.box),my(this_face.leye.best.box));

		cv::Point2f lthresholdm(lx(this_face.smile.best.box),my(this_face.smile.best.box));

		cv::Point2f rthresholdm(rx(this_face.smile.best.box),my(this_face.smile.best.box));

		temp_nose_list.clear();
		for (int N=0; N<this_face.nose.list.size(); ++N)
		{
			cv::Point2f nosemid(mxf(this_face.nose.list[N].box),myf(this_face.nose.list[N].box));
			float lthresholdn = lthresholde.x+(lthresholdm.x-lthresholde.x)*((nosemid.y-lthresholde.y)/(lthresholdm.y-lthresholde.y));
			float rthresholdn = rthresholde.x+(rthresholdm.x-rthresholde.x)*((nosemid.y-rthresholde.y)/(rthresholdm.y-rthresholde.y));
			if ((nosemid.x > lthresholdn) && (nosemid.x < rthresholdn))  temp_nose_list.push_back(this_face.nose.list[N]);
		}

		if ((temp_nose_list.size() > 0) && (temp_nose_list.size() < this_face.nose.list.size()))
		{
			this_face.nose.list = temp_nose_list;
		}
	}

	// if there is anything else left to do ...

	clogi("... nose count after pruning",this_face.nose.list.size());
}

//-----------------------------------------------------------------------------------------------------

void tidy_eyes_using_common_sense (FaceAndEyes& this_face)
{
    if ((this_face.leye.list.size() == 1) && (this_face.reye.list.size() == 1))  this_face.goodness = two_eyes_exact;
    else
    if ((this_face.leye.list.size() >= 1) && (this_face.reye.list.size() >= 1))  this_face.goodness = two_eyes;
    else
    if ((this_face.leye.list.size() >= 1) || (this_face.reye.list.size() >= 1))  this_face.goodness = one_eye;
    else
    if (this_face.nose.list.size() > 0)
    {
		if ((this_face.smile.list.size() >= 1) || (this_face.smile.lis2.size() >= 1))
			this_face.goodness = nose_and_mouth;
		else
			this_face.goodness = nose_only;
	}
	else
		this_face.goodness = nothing;

	this_face.process_this = true; // for the time being

	ResizeEyesInProportionToFace(this_face.leye.list,this_face.face.width);
	ResizeEyesInProportionToFace(this_face.reye.list,this_face.face.width);
	for (; merge_overlapping_eyes(this_face.leye.list); );
	for (; merge_overlapping_eyes(this_face.reye.list); );
	remove_wrong_eyes_which_overlap(this_face);
	PickBestReflections(this_face);
	estimate_missing_eye_by_reflection(this_face);
}

//-----------------------------------------------------------------------------------------------------

void tidy_noses_using_common_sense (FaceAndEyes& this_face)
{
	PruneNoseList(this_face);
	//ResizeNosesInProportionToFace(this_face.nose.list,this_face.face.width);
}

//-----------------------------------------------------------------------------------------------------
// When faces overlap, they can't both be right, so keep the best
// If several non-overlapping faces are detected, keep anything with eyes
// If no face has eyes but one has a nose and a smile or mouth, use that one

void prune_face_list (void)
{
	for (int i=0; i<faces_and_eyes_count; ++i)
	for (int j=i+1; j<faces_and_eyes_count; ++j)
	{
		if (o2(faces_and_eyes_list[i].face,faces_and_eyes_list[j].face))
		{
			if (faces_and_eyes_list[i].goodness < faces_and_eyes_list[j].goodness)  faces_and_eyes_list[i].process_this = false;
			if (faces_and_eyes_list[i].goodness > faces_and_eyes_list[j].goodness)  faces_and_eyes_list[j].process_this = false;
		}
	}

	int total_valid = 0;
	int total_semi = 0;

	for (int i=0; i<faces_and_eyes_count; ++i)  if (faces_and_eyes_list[i].process_this)
	{
		if (faces_and_eyes_list[i].goodness >= one_eye)
		{
			++total_valid;
		}
		else
		if (faces_and_eyes_list[i].goodness >= nose_and_mouth)
		{
			++total_semi;
		}
	}

	if (total_valid == 0)
	{
		// if we just have one detected face then work with that
		if (faces_and_eyes_count == 1)
		{
			faces_and_eyes_list[0].process_this = true;
		}
		else
		{
			for (int i=0; i<faces_and_eyes_count; ++i)  faces_and_eyes_list[i].process_this = (faces_and_eyes_list[i].goodness >= nose_and_mouth);
		}
	}
	else
	{
		for (int i=0; i<faces_and_eyes_count; ++i)  faces_and_eyes_list[i].process_this = (faces_and_eyes_list[i].goodness >= one_eye);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

bool qUseKnownFaces = false;

bool use_known_faces_if_available (FaceAndEyes& this_face)
{
	cv::Rect face;
	if (!is_known_face(face))  return false;
	this_face.face = face;
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Results logger

void log_prediction_results (const char *preamble, const cv::Point2i& known, const cv::Point2i& predicted)
{
	char buffer[100];
	int xerr,yerr;

	sprintf(buffer,"%s PREDICTION",preamble);
	clog2i(buffer,predicted);

	sprintf(buffer,"%s KNOWN",preamble);
	clog2i(buffer,known);

	sprintf(buffer,"%s ERROR X",preamble);
	clogi(buffer,xerr=100+abs(predicted.x-known.x));

	sprintf(buffer,"%s ERROR Y",preamble);
	clogi(buffer,yerr=100+abs(predicted.y-known.y));

	sprintf(buffer,"%s ERROR MAX",preamble);
	clogi(buffer,(xerr>yerr)?xerr:yerr);

	if (xerr < 106)
	{
		sprintf(buffer,"%s XERROR",preamble);
		clogi(buffer,predicted.x-known.x);
	}

	if (yerr < 106)
	{
		sprintf(buffer,"%s YERROR",preamble);
		clogi(buffer,predicted.y-known.y);
	}
}

void log_prediction_results (const char *preamble, const cv::Point2i& known, const cv::Point2f& predicted)
{
	cv::Point2i pi;
	pi.x = (int)predicted.x;
	pi.y = (int)predicted.y;
	log_prediction_results(preamble,known,pi);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

