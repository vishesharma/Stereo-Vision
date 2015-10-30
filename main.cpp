/* ************************************************************************
Oct. 10, 2015
Written by Vishesh Sharma

Right to use this code in any way you want without
warranty, support or any guarantee of it working.

USAGE GUIDE: 
Hardware: This program works with a stereo-camera setup. Two cameras 
(webcams) will be required. From now onwards, the words 'cameras' or 'webcams'
would be used interchangeably to refer to the stereo setup.

The program supports three modes of operation. 
- The first one is the 'normal' mode. In this mode, webcams are used for 
  environment sensing. 
- The second one is the 'test' mode. In this mode, cameras are not required
  since alreading existing data is used for processing.
- The third one is the 'capture' mode. In this mode, cameras are used to
  capture calibration and/or stereo image data for later testing (without 
  connecting the cameras again).

These modes can be used by setting the parameters correctly in the main()
function. Refer to main() for instructions to set the parameters.

----------------------------ACKNOWLEDGEMENT--------------------------------
The code for Stereo Calibration has been adapted from the following book:
Learning OpenCV: Computer Vision with the OpenCV Library
by Gary Bradski and Adrian Kaehler
Published by O'Reilly Media, October 3, 2008

AVAILABLE AT:
http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
Or: http://oreilly.com/catalog/9780596516130/
ISBN-10: 0596516134 or: ISBN-13: 978-0596516130
---------------------------------------------------------------------------

************************************************************************* */

#include "StereoVision.h"

#include <string>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

void main()
{
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////// Change these options for different operation modes. ///////////////////////////

	bool use_webcams = true;                     //Set to false to use existing data ('test' mode). 
												 //Must be true if calibration data 
												 //AND test data do not exist.

	int cam_left = 2;                            //Check port from your device manager
	int cam_right = 1;                           //Check port form your device manager

	bool capture_new_data = false;               //by default (set to true to run 'capture' mode)
	bool capture_new_calib_data = true;          //Change to false to collect new test data
	bool capture_new_test_data = !capture_new_calib_data;

	bool run_calibration = true;                //highly recommended

	
	int number_of_pictures_in_data = 0;          //Set to number of images in your test data

	string save_calib_data_to_folder = "";       //Change to path to folder where you want
												 //to store calibration data

	string load_calib_data_xml = "";             //Change to path to xml file containing complete 
												 //path of all calibration images

	string save_test_data_to_folder = "";        //Change to folder where you want to store test data
	string load_test_data_from_folder = save_test_data_to_folder;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	StereoVision NasusVision(cam_left, cam_right, run_calibration);   //For testing, use the alternate constructor

	// The alternate constructor
	//StereoVision NasusVision(run_calibration, load_calib_data_xml, load_test_data_from_folder, number_of_pictures_in_data);

	if (NasusVision.getCaptureNewData())
	{
		if (NasusVision.getCaptureNewCalibData())
		{
			NasusVision.captureMode(true, save_calib_data_to_folder, cam_left, cam_right, 13);
		}
		else
		{
			NasusVision.captureMode(false, save_test_data_to_folder, cam_left, cam_right, 13);
		}
	}

	if (run_calibration)
	{
		if (use_webcams)
		{
			NasusVision.captureMode(true, save_calib_data_to_folder, cam_left, cam_right, 13);
		}
		Size boardSize;
		bool showRectified = true;

		boardSize = Size(9, 6);              //Checkboard size for stereo calibration

		vector<string> image_list;
		bool ok = NasusVision.readStringList(load_calib_data_xml, image_list);
		if (!ok || image_list.empty())
		{
			cout << "can not open " << load_calib_data_xml << " or the string list is empty" << endl;
		}

		NasusVision.StereoCalib(image_list, boardSize, true, showRectified);
	}

	NasusVision.runStereoVision();
}