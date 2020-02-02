#include "roi_filter.hpp"

Rect Roi_filter::naive_roi(const Mat& img, unsigned int roi_size){
    if (roi_size == 0){
        //return img
        Rect R(Point(0,0), Point(frame_width, frame_height));
        return R;

    } else {

        Rect R(Point((frame_width/2)-(roi_size/2), (frame_height/2)-(roi_size/2)), Point((frame_width/2)+(roi_size/2), (frame_height/2)+(roi_size/2)));
        //return img(R);
        return R;
    }
}

Rect Roi_filter::real_roi(const Mat& img, unsigned int roi_size){
    if (roi_size == 0){
        //return img
        Rect R(Point(0,0), Point(frame_width, frame_height));
        return R;
    }

    //cv::Mat grey_mat = img.clone();
    cv::Mat grey_mat, grad_x, grad_y, abs_grad_x, abs_grad_y, sobel_mat, canny_mat;

    //Convert img to grey scale
    cv::cvtColor(img, grey_mat, CV_BGR2GRAY);

    //Blur img
    GaussianBlur(grey_mat, grey_mat, Size(3,3), 0, 0, BORDER_DEFAULT );

    //Sobel
    /// Gradient X
	Sobel( grey_mat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( grey_mat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_mat);

    //Canny
    int thresh = 100;
    Canny( sobel_mat, canny_mat, thresh, thresh*2);



    //Group contour
    int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
    vector<vector<Point> > contours;
    findContours(canny_mat, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        //minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
    }

    Rect outer_R = boundingRect(boundRect);
    return outer_R;

}