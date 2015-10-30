#include "StereoVision.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <Windows.h>

using namespace cv;
using namespace std;

// Normal mode of operation using the webcams
StereoVision::StereoVision(int webcam_left, int webcam_right, bool run_image_calibration)
{
	use_webcams = true;
	cam_left = webcam_left;
	cam_right = webcam_right;
	run_calib = run_image_calibration;
	capture_new_data = false;
}

// Testing using existing calibration and test stereo image data
StereoVision::StereoVision(bool run_calibration, string calibration_data_xml_folder = "", string stereo_image_data_folder = "", int number_pictures_in_test_data = 0)
{
	use_webcams = false;
	run_calib = run_calibration;
	calib_data_xml_folder = calibration_data_xml_folder;
	test_data_folder = stereo_image_data_folder;
	capture_new_data = false;
}


// StereoCalib() helper function for loading calibration data
bool StereoVision::readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}


// Function for stereo calibration 
void StereoVision::StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated, bool showRectified)
{
	Mat img_left, img_right, img_left_gray, img_right_gray;
	Mat imgDisparity16S, imgDisparity8U;
	const char *windowDisparity = "Disparity";
	int ndisparities_multiplier = 1;
	int ndisparities = 16 * 1;   /**< Range of disparity */
	int SADWindowSize = 11;      /**< Size of the block window. Must be odd */
	double minVal; double maxVal;

	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	bool displayCorners = true;
	const int maxScale = 2;
	const float squareSize = 1.f;  // Set this to your actual square size
	// ARRAY AND VECTOR STORAGE:

	vector<vector<Point2f>> imagePoints[2];
	vector<vector<Point3f>> objectPoints;

	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, 0);
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale);
				found = findChessboardCorners(timg, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 640. / MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf);
				imshow("corners", cimg1);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, E, F,
		CALIB_FIX_ASPECT_RATIO +
		CALIB_ZERO_TANGENT_DIST +
		CALIB_SAME_FOCAL_LENGTH +
		CALIB_RATIONAL_MODEL +
		CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "done with RMS error=" << rms << endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
				imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average reprojection err = " << err / npoints << endl;

	// save intrinsic parameters
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	// OpenCV can handle left-right
	// or up-down camera arrangements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

	// COMPUTE AND DISPLAY RECTIFICATION
	if (!showRectified)
		return;

	Mat rmap[2][2];
	// IF BY CALIBRATED (BOUGUET'S METHOD)
	if (useCalibrated)
	{
		// we already computed everything
	}
	// OR ELSE HARTLEY'S METHOD
	else
		// use intrinsic parameters of each camera, but
		// compute the rectification transformation directly
		// from the fundamental matrix
	{
		vector<Point2f> allimgpt[2];
		for (k = 0; k < 2; k++)
		{
			for (i = 0; i < nimages; i++)
				std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
		}
		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		Mat H1, H2;
		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

		R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
		R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
		P1 = cameraMatrix[0];
		P2 = cameraMatrix[1];
	}

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	/*Mat canvas;
	double sf;
	int w, h;*/
	/*if (!isVerticalStereo)
	{
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h, w * 2, CV_8UC3);
	}
	else
	{
	sf = 300. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h * 2, w, CV_8UC3);
	}*/

	for (i = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
			remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			/*Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);*/

			if (useCalibrated)
			{
				//cout << cvRound(validRoi[k].width) << endl << cvRound(validRoi[k].height);
				//Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf), cvRound(validRoi[k].width *sf), cvRound(validRoi[k].height*sf));

				//rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
				if (k == 0)
				{
					img_left = cimg;
					cvtColor(img_left, img_left_gray, CV_BGR2GRAY);
					for (j = 0; j < img_left_gray.rows; j += 16)
						line(img_left_gray, Point(0, j), Point(img_left_gray.cols, j), Scalar(0, 255, 0), 1, 8);
				}
				else
				{
					img_right = cimg; // canvas(vroi);
					cvtColor(img_right, img_right_gray, CV_BGR2GRAY);
					for (j = 0; j < img_right_gray.rows; j += 16)
						line(img_right_gray, Point(0, j), Point(img_right_gray.cols, j), Scalar(0, 255, 0), 1, 8);
				}
			}
		}
		ndisparities = 16 * 1;

		Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);

		sbm->compute(img_left_gray, img_right_gray, imgDisparity16S);

		minMaxLoc(imgDisparity16S, &minVal, &maxVal);

		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

		namedWindow("Right");
		namedWindow("Left");
		namedWindow(windowDisparity, WINDOW_NORMAL);

		imshow("Right", img_right_gray);
		imshow("Left", img_left_gray);
		imshow(windowDisparity, imgDisparity8U);
		//waitKey();
		/*if (!isVerticalStereo)
		for (j = 0; j < canvas.rows; j += 16)
		line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
		for (j = 0; j < canvas.cols; j += 16)
		line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);*/


		//imshow("rectified", canvas);
		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

// Captures calibration data and test data using
// the webcams nad stores them in the specified folder
void StereoVision::captureMode(bool calib_capture_mode, string save_to_folder, int cam_left, int cam_right, int num_pics_to_take)
{
	int num_pictures_taken = 0;
	bool take_picture;
	VideoCapture capture_left(cam_left);
	VideoCapture capture_right(cam_right);
	Mat image_left, image_right, progress;
	String text;

	for (;;)
	{
		if (num_pictures_taken == num_pics_to_take)
			break;

		progress = Mat::zeros(Size(400, 200), CV_8UC1);
		take_picture = false;

		capture_left >> image_left;
		capture_right >> image_right;

		flip(image_left, image_left, 1);
		flip(image_right, image_right, 1);

		text = "Press 'c' to take pictures. " + to_string(num_pictures_taken) + " pictures taken";
		putText(progress, text, Point(progress.cols / 10, progress.rows / 2), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255));

		namedWindow("Left");
		namedWindow("Right");
		namedWindow("Progress");

		imshow("Left", image_left);
		imshow("Right", image_right);
		imshow("Progress", progress);

		char input = waitKey(10);
		if (input == 'c')
			take_picture = true;
		else if (input == 27)
			return;

		if (take_picture == false)
			continue;

		if (take_picture)
		{
			string name_left;
			string name_right;

			if (calib_capture_mode)
			{
				name_left = save_to_folder + "/ImageLeft" + to_string(num_pictures_taken) + ".jpg";
				name_right = save_to_folder + "/ImageRight" + to_string(num_pictures_taken) + ".jpg";
			}
			else
			{
				name_left = save_to_folder + "/DataLeft" + to_string(num_pictures_taken) + ".jpg";
				name_right = save_to_folder + "/DataRight" + to_string(num_pictures_taken) + ".jpg";
			}

			imwrite(name_left, image_left);
			imwrite(name_right, image_right);

			num_pictures_taken++;
			cout << "taken\n";
		}
	}
}

// Runs the stereo-vision algorithm
void StereoVision::runStereoVision()
{
	vector<int> mask_contours_area;
	vector<vector<Point>> mask_contours;
	int Canny_threshold = 33;
	int SADWindowSize = 19; //(3 to 11)
	int numberOfDisparities = 16 * 11;
	int preFilterCap = 63;
	int minDisparity = 0;
	int uniquenessRatio = 10;
	int speckleWindowSize = 100; //(50 to 200)
	int speckleRange = 2; //( 1 or 2)
	int disp12MaxDiff = 1;
	bool fullDP = false;
	int P1_SGBM = 216;
	int P2_SGBM = 864;

	Mat R1, R2, P1, P2, Q;
	Mat rmap[2][2];
	Mat cameraMatrix[2], distCoeffs[2];
	Mat image_left, image_right, remapped_image_left, remapped_image_right, image_left_gray, image_right_gray, remapped_image_left_filtered;
	Mat eroded_Canny_image, abs_Sobel_image, filtered_Canny_image;
	Size imageSize;
	Mat imgDisparity16S, imgDisparity8U, Canny_image, contour_image, blob_image, mask_image, morphed_mask_image;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat Sobel_image_x, Sobel_image_y;
	Mat abs_Sobel_image_x, abs_Sobel_image_y;
	Mat density_scanner_image, density_image;
	Mat ROI_refined_mask;
	const char *windowDisparity = "Disparity";
	double minVal, maxVal;
	int num_data_sample = 0;

	const int MIN_CONTOUR_DISTANCE = 11;
	const int MAX_PROBABILITY = 100;
	const int MAX_CONSECUTIVE_ELEMS = 7;
	const int PROBABILITY_INCREMENT = 2;
	const int MIN_DETECTED_COMP = 750;
	const int ROI_THRESHOLD_CONTOUR_AREA = 35;        //25
	const int OBJECT_THRESHOLD_CONTOUR_AREA = 2000;
	const int THRESHOLD_CURVATURE = 4;
	const int SKIN_THRESHOLD_PERCENTAGE = 0.35;
	const int THRESHOLD_ROI_OBJECT_AREA_PERCENT = 0.15;

	struct properties
	{
		bool skin_color;
		bool cylindrical;
	};

	struct object
	{
		vector<Point> object_contour;
		vector<Point> object_hull;
		Vec3f hsv_vals;
		properties object_properties;
		float area;

		object(vector<Point> contour, vector<Point> hull, Vec3f hsv_color, float contour_area, bool is_skin)
		{
			object_hull = hull;
			object_contour = contour;
			hsv_vals = hsv_color;
			area = contour_area;
			object_properties.skin_color = is_skin;
			object_properties.cylindrical = false; //by default
		}
	};

	FileStorage fs("extrinsics.yml", FileStorage::READ);
	fs["R1"] >> R1;
	fs["R2"] >> R2;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs.release();

	fs.open("intrinsics.yml", FileStorage::READ);
	fs["M1"] >> cameraMatrix[0];
	fs["M2"] >> cameraMatrix[1];
	fs["D1"] >> distCoeffs[0];
	fs["D2"] >> distCoeffs[1];
	fs.release();

	VideoCapture capture_left;
	VideoCapture capture_right;

	for (int i = 0; i < 1; i++)
	{
		if (use_webcams)
		{
			capture_left.open(cam_left);
			capture_left >> image_left;
			capture_right.open(cam_right);
			capture_right >> image_right;
		}
		else
		{
			image_left = imread(test_data_folder + "/DataLeft0.jpg");
			image_right = imread(test_data_folder + "/DataRight0.jpg");
		}
		imageSize = image_left.size();

		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	}

	density_scanner_image = Mat::zeros(image_left.size(), CV_8UC1);
	uchar* pixel_ptr;

	for (int r = 30; r < density_scanner_image.rows - 30; r += 2)
	{
		pixel_ptr = density_scanner_image.ptr<uchar>(r);
		for (int c = 30; c < density_scanner_image.cols - 30; c += 2)
			pixel_ptr[c] = 255;
	}

	for (;;)
	{
		if (use_webcams)
		{
			capture_left >> image_left;
			capture_right >> image_right;
		}
		else
		{
			if (num_data_sample == number_of_pictures_in_dataset)
				return;
			image_left = imread(test_data_folder + "/DataLeft" + to_string(num_data_sample) + ".jpg");
			image_right = imread(test_data_folder + "/DataRight" + to_string(num_data_sample) + ".jpg");
			num_data_sample++;
		}

		remap(image_left, remapped_image_left, rmap[0][0], rmap[0][1], INTER_LINEAR);
		remap(image_right, remapped_image_right, rmap[1][0], rmap[1][1], INTER_LINEAR);
		cvtColor(remapped_image_left, image_left_gray, CV_BGR2GRAY);
		cvtColor(remapped_image_right, image_right_gray, CV_BGR2GRAY);

		//Generating the disparity image
		Ptr<StereoBM> sbm = StereoBM::create(numberOfDisparities, SADWindowSize);
		sbm->compute(image_left_gray, image_right_gray, imgDisparity16S);
		minMaxLoc(imgDisparity16S, &minVal, &maxVal);
		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

		//Alternative for generating disparity image (better result; much slower)
		/*
		Ptr<StereoSGBM> sbm = StereoSGBM::create(minDisparity, numberOfDisparities, SADWindowSize);
		sbm->compute(image_left_gray, image_right_gray, imgDisparity16S);
		minMaxLoc(imgDisparity16S, &minVal, &maxVal);
		imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
		*/

		threshold(imgDisparity8U, mask_image, 120, 255, 0);
		erode(mask_image, morphed_mask_image, getStructuringElement(MORPH_RECT, Size(3, 3)));
		dilate(morphed_mask_image, morphed_mask_image, getStructuringElement(MORPH_RECT, Size(3, 3)));
		threshold(morphed_mask_image, morphed_mask_image, 120, 255, 0);

		Mat reduced_zone_mask = Mat::zeros(morphed_mask_image.size(), CV_8UC1);
		rectangle(reduced_zone_mask, Rect(Point(30, 30), Point(morphed_mask_image.cols - 30, morphed_mask_image.rows - 30)), 255, -1);
		Mat reduced_morphed_mask_image;
		Mat reduced_morphed_mask_image_copy;
		morphed_mask_image.copyTo(reduced_morphed_mask_image, reduced_zone_mask);
		reduced_morphed_mask_image.copyTo(reduced_morphed_mask_image_copy);

		Mat skeleton_disparity_image(reduced_morphed_mask_image.size(), CV_8UC1, Scalar(0));
		Mat temp_disparity_image;
		Mat eroded_disparity_image;
		Mat disp_element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

		for (int i = 0; i < 5; i++)
		{
			erode(reduced_morphed_mask_image_copy, eroded_disparity_image, disp_element);
			dilate(eroded_disparity_image, temp_disparity_image, disp_element);
			subtract(reduced_morphed_mask_image_copy, temp_disparity_image, temp_disparity_image);
			bitwise_or(skeleton_disparity_image, temp_disparity_image, skeleton_disparity_image);
			eroded_disparity_image.copyTo(reduced_morphed_mask_image_copy);
		}

		int max_col_num = reduced_morphed_mask_image.cols - 2;
		int max_row_num = reduced_morphed_mask_image.rows - 2;
		int min_col_num = 2;			//To offset for ROI extraction
		int min_row_num = 2;			//To offset for ROI extraction

		Mat ROI_mask_image = Mat::zeros(morphed_mask_image.size(), CV_8UC1);
		Mat ROI_contour_image = Mat::zeros(remapped_image_left.size(), CV_8U);

		findContours(reduced_morphed_mask_image, mask_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		vector<Point2f> contour_centers;
		vector<Moments> contour_moments;
		vector<int> contour_areas;
		vector<int> contour_perimeters;
		int cur_contour_area;
		int cur_contour_perimeter;
		Moments cur_contour_moments;
		vector<bool> contour_of_interest;

		//Getting the area, perimeter and centers of the contours that are larger than the threshold area
		for (int k = 0; k < mask_contours.size(); k++)
		{
			cur_contour_area = contourArea(mask_contours[k]);
			if (cur_contour_area > ROI_THRESHOLD_CONTOUR_AREA)
			{
				contour_of_interest.push_back(true);
				contour_areas.push_back(cur_contour_area);
				contour_perimeters.push_back(arcLength(mask_contours[k], true));
				cur_contour_moments = moments(mask_contours[k]);
				contour_centers.push_back(Point2f(cur_contour_moments.m10 / cur_contour_moments.m00,
					cur_contour_moments.m01 / cur_contour_moments.m00));
			}
			else
			{
				contour_of_interest.push_back(false);
			}
		}

		vector<vector<Point>> hull(1);
		vector<RotatedRect> enclosing_rectangles;
		vector<Point> mask_rect_corners;
		Point2f rect_points[4];
		int min_dist, x, y, dist_between_contours;

		//Finding the enclosing rectangles
		for (int i = 0; i < contour_of_interest.size(); i++)
		{
			if (contour_of_interest[i])
			{
				enclosing_rectangles.push_back(minAreaRect(mask_contours[i]));
			}
		}

		int cur_contour_num = -1;
		int cur_approx_contour_radius;
		int num_neighbour_contours;

		//Finding the points that enclose the objects
		for (int i = 0; i < contour_areas.size(); i++)
		{
			int m;
			for (m = ++cur_contour_num; m < contour_of_interest.size(); m++)
			{
				if (contour_of_interest[m])
				{
					cur_contour_num = m;
					break;
				}
			}

			cur_contour_perimeter = contour_perimeters[i];
			cur_approx_contour_radius = cur_contour_perimeter / 5;

			min_dist = INT_MAX;
			num_neighbour_contours = 0;

			x = (int)contour_centers[i].x;
			y = (int)contour_centers[i].y;

			for (int l = 0; l < contour_areas.size(); l++)
			{
				if (l == i)
					continue;

				//Isolated small contours are considered imperfections 
				//produced in the disparity image and are discarded
				dist_between_contours = sqrt(pow((x - contour_centers[l].x), 2)
					+ pow((y - contour_centers[l].y), 2))
					- (cur_approx_contour_radius)
					-(contour_perimeters[l] / 5);

				if (dist_between_contours < MIN_CONTOUR_DISTANCE)
					num_neighbour_contours++;
			}

			if (num_neighbour_contours < 2)
				continue;

			enclosing_rectangles[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				mask_rect_corners.push_back((Point)rect_points[j]);
		}

		int max_x = 0, max_y = 0, min_x = INT_MAX, min_y = INT_MAX;
		bool found_object = false;

		//Finding the bounding upright rectangle for the object
		if (mask_rect_corners.size() > 10)
		{
			found_object = true;
			convexHull(mask_rect_corners, hull[0]);
			drawContours(ROI_mask_image, hull, 0, 255, -1);

			for (int i = 0; i < mask_rect_corners.size(); i++)
			{
				if (mask_rect_corners[i].x > max_x)
					max_x = mask_rect_corners[i].x;
				if (mask_rect_corners[i].x < min_x)
					min_x = mask_rect_corners[i].x;

				if (mask_rect_corners[i].y > max_y)
					max_y = mask_rect_corners[i].y;
				if (mask_rect_corners[i].y < min_y)
					min_y = mask_rect_corners[i].y;
			}

			if (max_x > max_col_num)
				max_x = max_col_num;

			if (max_y > max_row_num)
				max_y = max_row_num;

			if (min_x < min_col_num)
				min_x = min_col_num;

			if (min_y < min_row_num)
				min_y = min_row_num;
		}

		Mat ROI;
		Mat remapped_image_left_ROI = Mat::zeros(remapped_image_left.size(), remapped_image_left.type());
		Mat ROI_gray, ROI_Canny_image, ROI_morphed_Canny_image, channel[3];
		Mat ROI_hsv, ROI_hsv_threshold, ROI_hsv_threshold_modified;
		Mat contour_mask_image;
		Mat object_analysis_image;
		Mat contour_disparity_difference_image;
		Mat ROI_disparity_mask_image;

		imshow("Left", remapped_image_left);
		imshow("Right", remapped_image_right);

		if (found_object)
		{
			vector<vector<Point>> contours;
			vector<object> ROI_objects;
			vector<Point> ROI_points;

			remapped_image_left.copyTo(remapped_image_left_ROI, ROI_mask_image);

			ROI = remapped_image_left_ROI(Rect(Point(min_x - 2, min_y - 2), Point(max_x + 2, max_y + 2)));
			ROI_disparity_mask_image = reduced_morphed_mask_image_copy(Rect(Point(min_x - 2, min_y - 2), Point(max_x + 2, max_y + 2)));
			ROI_mask_image.copyTo(density_image, density_scanner_image);
			density_image = density_image(Rect(Point(min_x - 2, min_y - 2), Point(max_x + 2, max_y + 2)));

			for (int r = 0; r < density_image.rows; r += 2)
			{
				pixel_ptr = density_image.ptr<uchar>(r);
				for (int c = 0; c < density_image.cols; c += 2)
					if (pixel_ptr[c] == 255)
					{
						ROI_points.push_back(Point(c, r));
					}
			}

			int ROI_rows = ROI.rows, ROI_cols = ROI.cols;
			int x, y;

			cvtColor(ROI, ROI_gray, CV_BGR2GRAY);
			cvtColor(ROI, ROI_hsv, CV_BGR2HSV);

			Canny(ROI_gray, ROI_Canny_image, Canny_threshold, Canny_threshold * 2);

			ROI_morphed_Canny_image = Mat::zeros(ROI_Canny_image.size(), CV_8UC1);
			threshold(ROI_Canny_image, ROI_Canny_image, 100, 255, 0);
			dilate(ROI_Canny_image, ROI_morphed_Canny_image, getStructuringElement(MORPH_RECT, Size(3, 3)));
			findContours(ROI_morphed_Canny_image, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

			contour_image = Mat::zeros(ROI_Canny_image.size(), CV_8UC3);

			uchar* ROI_row_ptr;
			uchar* ROI_pixel_ptr;
			Vec3f color;
			float ROI_area = contourArea(contours[0]);
			float threshold_contour_area = THRESHOLD_ROI_OBJECT_AREA_PERCENT * ROI_area;
			float contour_area;


			for (int i = 1; i < contours.size(); i++)
			{
				contour_area = contourArea(contours[i]);

				if (contour_area > OBJECT_THRESHOLD_CONTOUR_AREA)
				{
					vector<Point> object_points;
					contour_mask_image = Mat::zeros(ROI.size(), CV_8UC1);
					drawContours(contour_mask_image, contours, i, 255, -1);

					contour_image = Mat::zeros(ROI.size(), CV_8UC1);///////////////////
					contour_disparity_difference_image = Mat::zeros(remapped_image_left.size(), CV_8UC1);

					drawContours(contour_image, contours, i, 255, -1);//////////////////
					subtract(contour_image, ROI_disparity_mask_image, contour_disparity_difference_image);
					cout << countNonZero(contour_disparity_difference_image) / contour_area << endl;

					imshow("MOD MASK", contour_mask_image);
					imshow("MASK", ROI_disparity_mask_image);
					imshow("DIFF", contour_disparity_difference_image);
					imshow("CONT", contour_image);///////////////////
					waitKey();//////////////////

					Mat skeleton_image(ROI.size(), CV_8UC1, Scalar(0));
					Mat temp_image;
					Mat eroded_image;
					Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

					for (int i = 0; i < 5; i++)
					{
						erode(contour_mask_image, eroded_image, element);
						dilate(eroded_image, temp_image, element); // temp = open(img)
						subtract(contour_mask_image, temp_image, temp_image);
						bitwise_or(skeleton_image, temp_image, skeleton_image);
						eroded_image.copyTo(contour_mask_image);
					}

					for (int r = 0; r < contour_mask_image.rows; r += 2)
					{
						pixel_ptr = contour_mask_image.ptr<uchar>(r);
						for (int c = 0; c < contour_mask_image.cols; c += 2)
							if (pixel_ptr[c] == 255)
							{
								object_points.push_back(Point(c, r));
							}
					}

					//blur(ROI_hsv, ROI_hsv, Size(3, 3));

					float avg_h_value = 0, avg_s_value = 0, avg_v_value = 0;
					int num_skin_colored_pixels = 0;

					for (int j = 0; j < object_points.size(); j++)
					{
						x = object_points[j].x;
						y = object_points[j].y;

						ROI_row_ptr = ROI_hsv.ptr(y);
						ROI_pixel_ptr = &ROI_row_ptr[3 * (x)];
						color.val[0] = (float)(*ROI_pixel_ptr);
						color.val[1] = (float)(*(++ROI_pixel_ptr));
						color.val[2] = (float)(*(++ROI_pixel_ptr));

						if (color[0] > 0 && color[0] < 30
							&& color[1] > 40 && color[1] < 120
							&& color[2] > 40 && color[2] < 140)
						{
							num_skin_colored_pixels++;
						}

						avg_h_value = ((avg_h_value * j) + color.val[0]) / (j + 1);
						avg_s_value = ((avg_s_value * j) + color.val[1]) / (j + 1);
						avg_v_value = ((avg_v_value * j) + color.val[2]) / (j + 1);
					}

					vector<Point> convex_hull;
					convexHull(contours[i], convex_hull);
					bool is_skin = false;
					if (((float)num_skin_colored_pixels / (float)object_points.size()) > SKIN_THRESHOLD_PERCENTAGE)
						is_skin = true;
					ROI_objects.push_back(object(contours[i], convex_hull, Vec3f(avg_h_value, avg_s_value, avg_v_value), contour_area, is_skin));
				}
			}

			vector<vector<Point>> cur_contour(1);
			Vec3f cur_color;
			Vec3f final_object_color;
			vector<vector<object>>::iterator final_objects_iter;
			vector<vector<object>> final_objects;

			for (vector<object>::iterator iter = ROI_objects.begin(); iter != ROI_objects.end(); iter++)
			{
				bool found_new = true;
				cur_contour[0] = iter->object_contour;
				cur_color = iter->hsv_vals;

				//cout << (float)cur_color[0] << endl << (float)cur_color[1] << endl << (float)cur_color[2] << endl << endl;

				for (final_objects_iter = final_objects.begin(); final_objects_iter != final_objects.end(); final_objects_iter++)
				{
					final_object_color = final_objects_iter[0][0].hsv_vals;

					if (abs(cur_color[0] - final_object_color[0]) < 2
						&& abs(cur_color[1] - final_object_color[1]) < 2
						&& abs(cur_color[2] - final_object_color[2]) < 2)
					{
						found_new = false;
						final_objects_iter[0].push_back(*iter);
						break;
					}
				}

				if (found_new)
				{
					vector<object> new_object;
					new_object.push_back(*iter);
					final_objects.push_back(new_object);
				}
			}

			Mat cur_image;
			vector<object>::iterator inner_iter;
			float object_area;
			Vec3f object_color;
			Rect object_boundary;
			bool is_skin;
			bool is_cylindrical = false;

			for (final_objects_iter = final_objects.begin(); final_objects_iter != final_objects.end(); final_objects_iter++)
			{
				object_area = 0;
				cur_image = Mat::zeros(ROI.size(), CV_8UC1);
				is_skin = final_objects_iter[0][0].object_properties.skin_color;

				for (inner_iter = final_objects_iter[0].begin(); inner_iter != final_objects_iter[0].end(); inner_iter++)
				{
					object_area += inner_iter->area;
					vector<vector<Point>> cur_contour(1);
					vector<vector<Point>> cur_hull(1);
					cur_contour[0] = inner_iter->object_contour;
					cur_hull[0] = inner_iter->object_hull;

					drawContours(cur_image, cur_contour, 0, 255, -1);

					float delta_x, delta_y;
					float slope, slope_prev;
					bool increasing_slope;
					int num_line = 0, num_consec_trend = 0;

					for (int i = 0; i < cur_hull[0].size() - 1; i++)
					{
						delta_x = cur_hull[0][i].x - cur_hull[0][i + 1].x;
						delta_y = cur_hull[0][i].y - cur_hull[0][i + 1].y;

						if (delta_x == 0) //The line is vertical
							continue;

						slope = delta_y / delta_x;

						if (slope < 1 && slope > -1) //The contour is non-vertical //USE MAJOR AXIS LATER
						{
							num_line++;

							if (num_line == 1)
							{
								slope_prev = slope;
								continue;
							}

							if (num_line == 2)
							{
								if (slope > slope_prev)
									increasing_slope = true;
								else
									increasing_slope = false;
								slope_prev = slope;
								continue;
							}

							if (slope > slope_prev && increasing_slope)
								num_consec_trend++;

							else if (slope < slope_prev && !increasing_slope)
								num_consec_trend++;

							else
							{
								num_line = 0;
								num_consec_trend = 0;
							}

							if (num_consec_trend > THRESHOLD_CURVATURE)
							{
								inner_iter->object_properties.cylindrical = true;
								is_cylindrical = true;
							}
							slope_prev = slope;
						}
					}
				}

				if (object_area > threshold_contour_area)
				{
					Scalar boundary_color;
					object_boundary = boundingRect(final_objects_iter[0][0].object_contour);
					int rect_x1 = object_boundary.x;
					int rect_y1 = object_boundary.y;
					int rect_x2 = rect_x1 + object_boundary.width;
					int rect_y2 = rect_y1 + object_boundary.height;

					if (is_skin)
					{
						boundary_color = Scalar(0, 0, 255);
					}
					else
					{
						boundary_color = Scalar(255, 0, 0);
					}

					rectangle(remapped_image_left,
						Point(min_x + rect_x1, min_y + rect_y1),
						Point(min_x + rect_x2, min_y + rect_y2), boundary_color, 2);

					if (is_cylindrical && !is_skin)
					{
						putText(remapped_image_left, "Cylindrical",
							Point(min_x + object_boundary.x, min_y + object_boundary.y + 10), 1, 1, Scalar(0, 255, 0));
					}
				}
			}
			imshow("Final", remapped_image_left);
			//imshow("ROI", ROI);
		}

		//imshow("DISP", imgDisparity8U);

		char c = waitKey();

		if (c == 27)
			return;

		/*if (c == 110)
		num_picture++;*/
	}
}