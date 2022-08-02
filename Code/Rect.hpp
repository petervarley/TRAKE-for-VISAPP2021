#ifndef _RECT_HPP_
#define _RECT_HPP_

#include <opencv2/core/core.hpp>

int lx (const cv::Rect& rect);
int mx (const cv::Rect& rect);
float mxf (const cv::Rect& rect);
int rx (const cv::Rect& rect);

int ty (const cv::Rect& rect);
int my (const cv::Rect& rect);
float myf (const cv::Rect& rect);
int by (const cv::Rect& rect);

int ho (const cv::Rect& P, const cv::Rect& Q);
int vo (const cv::Rect& P, const cv::Rect& Q);
int o2 (const cv::Rect& P, const cv::Rect& Q);

#endif
