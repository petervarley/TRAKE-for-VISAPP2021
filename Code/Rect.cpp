
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include "Rect.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////

int lx (const cv::Rect& rect)
{
	return rect.x;
}

int mx (const cv::Rect& rect)
{
	return rect.x + (rect.width/2);
}

float mxf (const cv::Rect& rect)
{
	return rect.x + ((float)rect.width/2.0);
}

int rx (const cv::Rect& rect)
{
	return rect.x + rect.width-1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

int ty (const cv::Rect& rect)
{
	return rect.y;
}

int my (const cv::Rect& rect)
{
	return rect.y + (rect.height/2);
}

float myf (const cv::Rect& rect)
{
	return rect.y + ((float)rect.height/2.0);
}

int by (const cv::Rect& rect)
{
	return rect.y + rect.height-1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

int ho (const cv::Rect& P, const cv::Rect& Q)
{
	int pl = lx(P);
	int pr = rx(P);
	int ql = lx(Q);
	int qr = rx(Q);

	if (pr <= ql)  return 0;
	if (qr <= pl)  return 0;

	int po = pr-ql;
	int qo = qr-pl;

	return (po < qo) ? po : qo;
}

//-----------------------------------------------------------------------------------------------------

int vo (const cv::Rect& P, const cv::Rect& Q)
{
	int pt = ty(P);
	int pb = by(P);
	int qt = ty(Q);
	int qb = by(Q);

	if (pb <= qt)  return 0;
	if (qb <= pt)  return 0;

	int po = pb-qt;
	int qo = qb-pt;

	return (po < qo) ? po : qo;
}

//-----------------------------------------------------------------------------------------------------

int o2 (const cv::Rect& P, const cv::Rect& Q)
{
	int h = ho(P,Q);
	int v = vo(P,Q);
	return (h < v) ? h : v;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
