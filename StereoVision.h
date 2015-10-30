#ifndef STEREOVISION_H
#define STEREOVISION_H

#include "opencv2/opencv.hpp"

#include <string>
#include <vector>

class StereoVision
{
public:
	StereoVision(int webcam_left, int webcam_right, bool run_calibration);
	StereoVision(bool run_calibration, std::string load_calibration_data_from_xml, std::string load_test_data_from_folder, int number_of_pictures_in_data);

	static void StereoCalib(const std::vector<std::string>& imagelist, cv::Size boardSize, bool useCalibrated = true, bool showRectified = true);
	static bool readStringList(const std::string& filename, std::vector<std::string>& l);

	void captureMode(bool calib_capture_mode, std::string save_to_folder, int cam_left, int cam_right, int num_pics_to_take);
	void runStereoVision();

	inline bool getUseWebCams() { return use_webcams; }
	inline bool getCaptureNewData() { return capture_new_data; }
	inline bool getCaptureNewCalibData() { return capture_new_calib_data; }
	inline bool getCaptureNewTestData() { return capture_new_test_data; }
	inline bool getRunCalib() { return run_calib; }

private:
	bool use_webcams;
	bool capture_new_data;
	bool capture_new_calib_data;
	bool capture_new_test_data;
	bool run_calib;

	int cam_left;
	int cam_right;
	int number_of_pictures_in_dataset;

	std::string calib_data_xml_folder;
	std::string test_data_folder;
};
#endif //STEREOVISION_H