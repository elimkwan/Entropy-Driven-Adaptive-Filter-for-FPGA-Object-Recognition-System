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

Rect Roi_filter::real_roi(const Mat& img, unsigned int roi_size){

    //cv::Mat grey_mat = img.clone();
    cv::Mat grey_mat, grad_x, grad_y, abs_grad_x, abs_grad_y, sobel_mat, canny_mat;
    int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

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
    double max_area = 0;
    vector<vector<Point> > contours;
    findContours(canny_mat, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    
    if (contours.size() <= 0){
        //too dark cant extract any contours
        Rect R(Point(0,0), Point(frame_width, frame_height));
        return R;
    }

    vector<vector<Point> > contours_poly(contours.size());
    //vector<vector<Point> > potential_contours;
    Rect boundRect;
    //vector<Point2f>centers( contours.size() );
    //vector<float>radius( contours.size() );
    int k = 0;

    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );

        // double m = contourArea(contours_poly[i]);
        // if (m > max_area){
        //     max_area = m;
        //     boundRect = boundingRect(contours_poly[i]);

        // }
        //cout << "loop at all?" <<endl;
        k = i;
    }

    //sort contour by size: ascending 
    //cout << "sorting contours" <<endl;
    std::sort(contours_poly.begin(), contours_poly.end(), contour_sorter());
    
    //cout << "sorting contours x2" <<endl;
    //cout << "what is k: " << k << endl;
    Rect max_r = boundingRect(contours_poly[k]);
    //cout << "sorting contours x2 debug" <<endl;
    double m = contourArea(contours_poly[k]);
    //cout << "sorting contours x2 debug debug" <<endl;
    //potential_contours.push_back(contours_poly[contours.size()-1]);
    //start from the second largest rect, loop till the smallest
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

            //potential_contours.push_back(contours_poly[i]);
            //cout << "sorting contours x4" <<endl;
        }
    }

    Rect wrapper = expand_r(x1,y1,x2,y2,0.1);
    //cout << "sorting contours x5" <<endl;

    // rectangle(grey_mat, wrapper, Scalar(0, 0, 255));

    // imshow( "sobel", sobel_mat);
	// waitKey(0);
	// imshow( "canny", canny_mat);
	// waitKey(0);
    // imshow( "grey", grey_mat);
	// waitKey(0);

    // vector<int> compression_params;
    // compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
    // compression_params.push_back( 100 );
    // std::string img_path = "./test_results/canny.jpg";
    // imwrite(img_path,canny_mat, compression_params);

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