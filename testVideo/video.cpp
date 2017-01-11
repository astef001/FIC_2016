#include <sstream>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
//#include <opencv2\highgui.h>
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2\cv.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const std::string windowName = "Original Image";
const std::string windowName1 = "HSV Image";
const std::string windowName2 = "Thresholded Image";
const std::string windowName3 = "After Morphological Operations";
const std::string trackbarWindowName = "Trackbars";


void on_mouse(int e, int x, int y, int d, void *ptr)
{
	if (e == EVENT_LBUTTONDOWN)
	{
		cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
	}
}

void on_trackbar(int, void*)
{//This function gets called whenever a
 // trackbar position is changed
}

string intToString(int number) {


	std::stringstream ss;
	ss << number;
	return ss.str();
}

void createTrackbars() {
	//create window for trackbars


	namedWindow(trackbarWindowName, 0);
	//create memory to store trackbar name on window
	char TrackbarName[50];
	sprintf(TrackbarName, "H_MIN", H_MIN);
	sprintf(TrackbarName, "H_MAX", H_MAX);
	sprintf(TrackbarName, "S_MIN", S_MIN);
	sprintf(TrackbarName, "S_MAX", S_MAX);
	sprintf(TrackbarName, "V_MIN", V_MIN);
	sprintf(TrackbarName, "V_MAX", V_MAX);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
	//the max value the trackbar can move (eg. H_HIGH),
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar);


}
void drawObject(int x, int y, Mat &frame) {

	//use some of the openCV drawing functions to draw crosshairs
	//on your tracked image!

	//UPDATE:JUNE 18TH, 2013
	//added 'if' and 'else' statements to prevent
	//memory errors from writing off the screen (ie. (-25,-25) is not within the window!)

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25 > 0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25 < FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25 > 0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25 < FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, intToString(x) + "," + intToString(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);
	//cout << "x,y: " << x << ", " << y;

}
void morphOps(Mat &thresh) {

	//create structuring element that will be used to "dilate" and "erode" image.
	//the element chosen here is a 3px by 3px rectangle

	Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate with larger element so make sure object is nicely visible
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));

	erode(thresh, thresh, erodeElement);
	erode(thresh, thresh, erodeElement);


	dilate(thresh, thresh, dilateElement);
	dilate(thresh, thresh, dilateElement);



}
void trackFilteredObject(int &x, int &y, Mat threshold, Mat &cameraFeed) {

	Mat temp;
	threshold.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//use moments method to find our filtered object
	double refArea = 0;
	bool objectFound = false;
	if (hierarchy.size() > 0) {
		int numObjects = hierarchy.size();
		//if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
		if (numObjects < MAX_NUM_OBJECTS) {
			for (int index = 0; index >= 0; index = hierarchy[index][0]) {

				Moments moment = moments((cv::Mat)contours[index]);
				double area = moment.m00;

				//if the area is less than 20 px by 20px then it is probably just noise
				//if the area is the same as the 3/2 of the image size, probably just a bad filter
				//we only want the object with the largest area so we safe a reference area each
				//iteration and compare it to the area in the next iteration.
				if (area > MIN_OBJECT_AREA && area<MAX_OBJECT_AREA && area>refArea) {
					x = moment.m10 / area;
					y = moment.m01 / area;
					objectFound = true;
					refArea = area;
				}
				else objectFound = false;


			}
			//let user know you found an object
			if (objectFound == true) {
				putText(cameraFeed, "Tracking Object", Point(0, 50), 2, 1, Scalar(0, 255, 0), 2);
				//draw object location on screen
				//cout << x << "," << y;
				drawObject(x, y, cameraFeed);

			}


		}
		else putText(cameraFeed, "TOO MUCH NOISE! ADJUST FILTER", Point(0, 50), 1, 2, Scalar(0, 0, 255), 2);
	}
}

struct RobotColor {
	Scalar min, max;
	std::string name;
};

static const std::pair<Scalar, Scalar> colors[] = {
	{ Scalar(50, 208, 99), Scalar(111, 256, 256) }, // blue
	{ Scalar(44, 244, 57), Scalar(80, 256, 256) }, // green
	{ Scalar(0, 221, 190), Scalar(29, 256, 256) }, // red
	// { Scalar(0, 41, 128), Scalar(254, 66, 256) }, // red
	{ Scalar(16, 227, 184), Scalar(77, 256, 256) }, // yellow
};

static const RobotColor YELLOW { colors[3].first, colors[3].second, "yellow" };
static const RobotColor BLUE { colors[0].first, colors[0].second, "blue" };
static const RobotColor GREEN { colors[1].first, colors[1].second, "green" };
static const RobotColor RED { colors[2].first, colors[2].second, "red" };

static const RobotColor BLACK { Scalar(0, 240, 0), Scalar(256, 256, 20), "black" };


//some boolean variables for different functionality within this
//program
bool trackObjects = true;
bool useMorphOps = true;

void getObjectPosition(int &x, int &y, const Mat& HSV, Mat& cameraFeed, const RobotColor& color)
{
	Mat threshold;
	//filter HSV image between values and store filtered image to
	//threshold matrix
	inRange(HSV, color.min, color.max, threshold);

	//perform morphological operations on thresholded image to eliminate noise
	//and emphasize the filtered object(s)
	if (useMorphOps)
		morphOps(threshold);

	//pass in thresholded frame to our object tracking function
	//this function will return the x and y coordinates of the
	//filtered object
	if (trackObjects) {
		trackFilteredObject(x, y, threshold, cameraFeed);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		printf("Usage: %s <rtmp source>\n", argv[0]);
		return 1;
	}

	Point p;
	//Matrix to store each frame of the webcam feed
	Mat cameraFeed;
	//matrix storage for HSV image
	Mat HSV;
	//matrix storage for binary threshold image
	Mat threshold;
	//x and y values for the location of the object
	int x = 0, y = 0;
	//create slider bars for HSV filtering
	createTrackbars();
	//video capture object to acquire webcam feed
	VideoCapture capture;
	//open capture object at location zero (default location for webcam)
	printf("Opening stream %s...\n", argv[1]);
	capture.open("rtmp://172.16.254.63/live/live");
	//set height and width of capture frame
	capture.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
	//start an infinite loop where webcam feed is copied to cameraFeed matrix
	//all of our operations will be performed within this loop

	std::cout << "Connecting to robot..\n";
	int robot = socket(AF_INET, SOCK_STREAM, 0);
	if (robot == -1) {
		perror("socket failed");
	}
	struct sockaddr_in robotAddress;
	robotAddress.sin_family = AF_INET;
	robotAddress.sin_port = htons(20231);
	inet_aton("193.226.12.217", &robotAddress.sin_addr);
	memset(robotAddress.sin_zero, 0, sizeof(robotAddress.sin_zero));
	std::cout << "created socket; connecting..." << std::endl;
	if (connect(robot, (struct sockaddr*) &robotAddress, sizeof(robotAddress)) != 0) {
		perror("error connecting to robot");
	}
	/*std::cout << "sending right...\n";
	if (send(robot, "r", 1, 0) != 1) {
		perror("error sending command");
	}
	sleep(1);
	/*if (send(robot, "s", 1, 0) != 1) {
		perror("error sending command");
	}
	std::cout << "sent stop...\n";
*/
	RobotColor primary = RED, secondary = GREEN;

	while (1) {
		//store image to matrix
		capture.read(cameraFeed);
		if (cameraFeed.empty()) {
			std::cout << "feed empty" << std::endl;
			continue;
		}


		//convert frame from BGR to HSV colorspace
		cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);

		int px, py, sx, sy;
		getObjectPosition(px, py, HSV, cameraFeed, primary);
		getObjectPosition(sx, sy, HSV, cameraFeed, secondary);
		int tx, ty;
		getObjectPosition(tx, ty, HSV, cameraFeed, BLUE);
		int orientation_axis = int(atan(((double) sy - py)/(sx - px))  * 180 / 3.14156);
		int orientation_enemy = int(atan(((double) ty - py)/(tx - px))  * 180 / 3.14156);
		std::cout << "Position: x = " << px << " y = " << py << "\n" << std::endl;
		std::cout << "Orientation friendly: " << orientation_axis << " Orientation enemy: " << orientation_enemy << std::endl;
		std::cout << "\nPositionEnemy: x = " << tx << " y = " << ty << "\n" << std::endl;
		//int det_Col = px*sy + sx*ty + py*tx - tx*sy - sx*py - ty*px;
		if(orientation_axis != orientation_enemy){
			send(robot, "l", 1, 0);
			send(robot, "s", 1, 0);
		}
		else{
			send(robot, "s", 1, 0);
			send(robot, "f", 1, 0);
		}
		//show frames
		imshow(windowName, cameraFeed);
		setMouseCallback("Original Image", on_mouse, &p);
		//delay 30ms so that screen can refresh.
		//image will not appear without this waitKey() command
		waitKey(30);
	}

	return 0;
}
