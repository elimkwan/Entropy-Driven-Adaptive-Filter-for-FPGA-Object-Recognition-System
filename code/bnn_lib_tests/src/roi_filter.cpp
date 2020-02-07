#include "roi_filter.hpp"

// compare contour
struct contour_sorter {
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect ra(boundingRect(a));
        Rect rb(boundingRect(b));
        // scale factor for y should be larger than img.width
        return ( (ra.x + 1000*ra.y) < (rb.x + 1000*rb.y) );
    }
};

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


Rect Roi_filter::basic_roi(const Mat& cur_mat){

    cv::Mat grey_mat, grad_x, grad_y, abs_grad_x, abs_grad_y, sobel_mat, canny_mat;
    int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

    //Convert img to grey scale
    cv::cvtColor(cur_mat, grey_mat, CV_BGR2GRAY);

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
    double max_area = 0;
    vector<vector<Point> > contours;
    findContours(canny_mat, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.size() <= 0){
        //too dark cant extract any contours
        Rect R(Point(0,0), Point(frame_width, frame_height));
        return R;
    }

    vector<vector<Point> > contours_poly(contours.size());
    Rect boundRect;

    int k = 0;

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        k = i;
    }

    std::sort(contours_poly.begin(), contours_poly.end(), contour_sorter());
    
    Rect max_r = boundingRect(contours_poly[k]);

    double m = contourArea(contours_poly[k]);

    int x1,y1,x2,y2;
    x1 = max_r.x;
    y1 = max_r.y;
    x2 = max_r.x + max_r.width;
    y2 = max_r.y + max_r.height;
    for(size_t i = k-2 ; i > 0; i--){
        double a = contourArea(contours_poly[i]);
        //cout << "sorting contours x3" <<endl;
        if (a > (m/2)){
            Rect r1 = boundingRect(contours_poly[i]);
            x1 = min(x1,r1.x);
            y1 = min(y1,r1.y);
            x2 = max(x2,r1.x + r1.width);
            y2 = max(y2,r1.y + r1.height);
        }
    }

    Rect wrapper = expand_r(x1,y1,x2,y2,0.1);

    return wrapper;

}

Rect Roi_filter::expand_r(int x1, int y1, int x2, int y2, float p){
    
    int a = x1 - x1*p;
    int b = y1 - y1*p;
    int c = x2 + x2*p;
    int d = y2 + y2*p;

    if ( a < 0 || a > frame_width){
        a = 0;
    }
    if ( b < 0 || b > frame_height){
        b = 0;
    }

    if ( c < 0 || c > frame_width || c < a){
        c = frame_width;
    }

    if ( d < 0 || d > frame_height || d < b){
        d = frame_height;
    }

    Rect R(Point(a,b), Point(c, d));

    return R;
}

Rect Roi_filter::enhanced_roi (const Mat& img){
    cv::Mat grey;
    cur_mat = img.clone();
    cvtColor(img, grey, COLOR_BGR2GRAY);
    cur_mat_grey = grey.clone();

    cv::Mat contour_mat, motion_mat, weighted_mat;
    Rect bounding_r;
    
    cout<<"breakpoint1"<<endl;
    contour_mat = contour_map();
    cout<<"breakpoint2"<<endl;

    motion_mat = dense_optical_flow(contour_mat);
    cout<<"breakpoint3"<<endl;

    weighted_mat = weighted_map(motion_mat);
    cout<<"breakpoint4"<<endl;

    bounding_r = bounding_rect(weighted_mat);
    cout<<"breakpoint5"<<endl;

    update_enhanced_roi_param(motion_mat);
    cout<<"breakpoint6"<<endl;

    return bounding_r;
    
}

void Roi_filter::init_enhanced_roi(const Mat& img){
    cv::Mat grey_mat;
    Mat M(frame_width, frame_height, CV_8UC3, Scalar(0,0,0));

    cout<<"initialising"<<endl;

    prev_mat = img.clone();
    cvtColor(img,grey_mat, COLOR_BGR2GRAY);
    prev_mat_grey = grey_mat.clone();
    mask1 = M.clone();
    mask2 = M.clone();
    mask3 = M.clone();
    cout<<"sucessfully init"<<endl;
}

void Roi_filter::update_enhanced_roi_param (const Mat& motion_mat){
 
    prev_mat = cur_mat.clone();
    prev_mat_grey = cur_mat_grey.clone();
    mask3 = mask2.clone();
    mask2 = mask1.clone();
    mask1 = motion_mat.clone();

}

cv::Mat Roi_filter::contour_map(){
    cv::Mat a,blur_mat,grad_x, grad_y, abs_grad_x, abs_grad_y, sobel_mat, canny_mat;
    cv::Mat contour_mat = Mat::zeros(cur_mat.rows, cur_mat.cols, CV_8UC3);

    cout<<"breakpoint1.1"<<endl;
    a = cur_mat_grey.clone();
    cout<<"breakpoint1.2"<<endl;
    int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

    //Blur img
    GaussianBlur(a, blur_mat, Size(3,3), 0, 0, BORDER_DEFAULT );

    //Sobel
    /// Gradient X
	Sobel(blur_mat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( blur_mat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_mat);

    //Canny
    int thresh = 100;
    Canny( sobel_mat, canny_mat, thresh, thresh*2);

    //Group contour
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_mat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    cout<<"breakpoint1.3"<<endl;
    if (contours.size() <= 0){
        //too dark cant extract any contours, return black img
        cout << "too dark contour map empty" <<endl;
        Mat M(frame_width, frame_height, CV_8UC3, Scalar(0,0,0));
        cout<<"breakpoint1.4"<<endl;
        //cout << "Contour Black = " << endl << " " << M << endl << endl;
        cout<<"breakpoint1.5"<<endl;
        return M;
    }

    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours(contour_mat, contours, idx, color, CV_FILLED, 8, hierarchy );
    }

    if (contour_mat.empty()){
        cout << "contour map empty" <<endl;
    } else {
        //cout << "Contour = " << endl << " " << contour_mat << endl << endl;
        imshow("Contour Mat", contour_mat);
        waitKey(0);
    }

    return contour_mat;
}

cv::Mat Roi_filter::dense_optical_flow(const Mat& contour_mat){

    cv::Mat cur, prev;
    cur = cur_mat_grey.clone();
    prev = prev_mat_grey.clone();

    Mat flow(prev.size(), CV_32FC2);
    calcOpticalFlowFarneback(prev, cur, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm, magn_combined;
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);


    //combining contour map with motion map
    add(magnitude, contour_mat, magn_combined);

    normalize(magn_combined, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));

    //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle; //hue
    _hsv[1] = Mat::ones(angle.size(), CV_32F); //saturation
    _hsv[2] = magn_norm; //value
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, COLOR_HSV2BGR);

    if (bgr.empty()){
        cout << "motion map is empty" <<endl;
    } else {
        //cout << "Optical = " << endl << " " << bgr << endl << endl;
        imshow("Dense optical flow", bgr);
        waitKey(0);
    }

    return bgr;
}

cv::Mat Roi_filter::weighted_map(const Mat& cur){

    cv::Mat m1,m2,m3;
    float a, b1, b2, b3;
    a=1.0;
    b1=0.5;
    b2=0.25;
    b3=0.125;

    addWeighted(cur, a, mask1, b1, 0.0, m1);
    addWeighted(m1, a, mask2, b2, 0.0, m2);
    addWeighted(m2, a, mask3, b3, 0.0, m3);

    return m3;
}

Rect Roi_filter::bounding_rect(const Mat& cur){

    //Canny
    cv::Mat canny_mat;
    int thresh = 100;

    Canny(cur, canny_mat, thresh, thresh*2);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_mat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    if (contours.size() <= 0){
        //too dark cant extract any contours
        cout << "too dark cant compute bounding rect"<<endl;
        Rect R(Point(0,0), Point(frame_width, frame_height));
        return R;
    }

    vector<vector<Point> > contours_poly(contours.size());
    Rect boundRect;
    int k = 0;

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        k = i;
    }

    std::sort(contours_poly.begin(), contours_poly.end(), contour_sorter());
    
    Rect max_r = boundingRect(contours_poly[k]);

    return max_r;
}