/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/xf_headers.hpp"

#include "xf_canny_config.h"
#include "xf_houghlines_config.h"

#include "xcl2.hpp"


typedef unsigned char NMSTYPE;

void AverageGaussian(cv::Mat& src, cv::Mat& dst);

void xiHoughLinesstandard(cv::Mat& img,
                          std::vector<cv::Vec2f>& lines,
                          float rho,
                          float theta,
                          int threshold,
                          int linesMax,
                          int maxtheta,
                          int mintheta);

int main(int argc, char** argv) {
    //# Images
    cv::Mat in_img;
    cv::Mat img_gray, hls_img, ocv_img, out_img, out_img_edge;
    cv::Mat diff;

    if (argc != 2) {
        printf("Usage : <executable> <input image> \n");
        return -1;
    }

    char img_path[1000];

    in_img = cv::imread(argv[1], 1); // reading in the color image
    if (!in_img.data) {
        printf("Failed to load the image ... %s\n!", argv[1]);
        return -1;
    }

    extractChannel(in_img, img_gray, 1); // Extract gray scale image

    hls_img.create(img_gray.rows, img_gray.cols, img_gray.depth());      // HLS image creation
    out_img.create(img_gray.rows, img_gray.cols / 4, img_gray.depth());  // HLS image creation
    out_img_edge.create(img_gray.rows, img_gray.cols, img_gray.depth()); // HLS image creation

    int height, width;
    int low_threshold, high_threshold;
    height = img_gray.rows;
    width = img_gray.cols;
    low_threshold = 30;
    high_threshold = 64;

    //////////////////////////////////////////////////////CL///////////////////////////////////

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_canny");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "canny_accel");

    std::vector<cl::Memory> inBufVec, outBufVec;
    cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer imageFromDevice(context, CL_MEM_READ_WRITE, (height * width / 4));

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
    krnl.setArg(2, height);
    krnl.setArg(3, width);
    krnl.setArg(4, low_threshold);
    krnl.setArg(5, high_threshold);

    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * (width)), img_gray.data);
    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    printf("before kernel - canny");
    // Launch the kernel
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    printf("after kernel - canny");

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    // q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height*width/4), out_img.data);
    // q.finish();

    cl::Kernel krnl2(program, "edgetracing_accel");
    // cl::Buffer imageToDeviceedge(context, CL_MEM_READ_WRITE,(height*width/4));
    cl::Buffer imageFromDeviceedge(context, CL_MEM_WRITE_ONLY, (height * width));

    // Set the kernel arguments
    krnl2.setArg(0, imageFromDevice);
    krnl2.setArg(1, imageFromDeviceedge);
    krnl2.setArg(2, height);
    krnl2.setArg(3, width);

    // q.enqueueWriteBuffer(imageToDeviceedge, CL_TRUE, 0, (height*(width/4)), out_img.data);

    // Profiling Objects
    cl_ulong startedge = 0;
    cl_ulong endedge = 0;
    double diff_prof_edge = 0.0f;
    cl::Event event_sp_edge;

    printf("before kernel - edge tracing");
    // Launch the kernel
    q.enqueueTask(krnl2, NULL, &event_sp_edge);
    clWaitForEvents(1, (const cl_event*)&event_sp_edge);

    printf("after kernel - edge tracing");

    event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_START, &startedge);
    event_sp_edge.getProfilingInfo(CL_PROFILING_COMMAND_END, &endedge);
    diff_prof_edge = endedge - startedge;
    std::cout << "\n Profiling edge tracing " << (diff_prof_edge / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    q.enqueueReadBuffer(imageFromDeviceedge, CL_TRUE, 0, (height * width), out_img_edge.data);

    q.finish();

    /////////////////////////////////end of CL call//////////////////////////////////////////////////

    /*				Apply Gaussian mask and call opencv canny function				*/
    cv::Mat img_gray1;
    img_gray1.create(img_gray.rows, img_gray.cols, img_gray.depth());
    AverageGaussian(img_gray, img_gray1); // Gaussian filter

#if L1NORM
    cv::Canny(img_gray1, ocv_img, 30.0, 64.0, FILTER_WIDTH, false); // Opencv canny function

#else
    cv::Canny(img_gray1, ocv_img, 30.0, 64.0, FILTER_WIDTH, true); // Opencv canny function
#endif

    absdiff(ocv_img, out_img_edge, diff); // Absolute difference between opencv and hls result
    imwrite("hls.png", out_img_edge);     // Save HLS result
    imwrite("ocv.png", ocv_img);          // Save Opencv result
    imwrite("diff.png", diff);
    // Save difference image
    // Find minimum and maximum differences.
    double minval = 256, maxval = 0;

    int cnt = 0;
    for (int i = 0; i < diff.rows - 0; i++) {
        for (int j = 0; j < diff.cols - 0; j++) {
            uchar v = diff.at<uchar>(i, j);

            if (v > 0) cnt++;
            if (minval > v) minval = v;
            if (maxval < v) maxval = v;
        }
    }

    float err_per = 100.0 * (float)cnt / (diff.rows * diff.cols);

    fprintf(stderr,
            "Minimum error in intensity = %f\n Maximum error in intensity = %f\n Percentage of pixels above error "
            "threshold = %f\nNo of Pixels with Error = %d\n",
            minval, maxval, err_per, cnt);

    fprintf(stderr, "kernel done - edge tracing");
    // if (err_per > 2.5f) return 1;
    ///////////////////////////////// HW HOUGH TRANSFORM //////////////////////////////////////////////////
    // Running the HLS accelerated Hough Transform
    printf("before kernel - houghlines");

    cl::Kernel krnl3(program, "houghlines_accel");

    // OpenCL section:
    size_t image_in_size_bytes = img_gray.rows * img_gray.cols * sizeof(unsigned char);
    size_t image_out_size_bytes = LINESMAX * sizeof(float);

    std::vector<float> outputrho(LINESMAX);
    std::vector<float> outputtheta(LINESMAX);

    cl::Buffer buffer_hough_inImage(context, CL_MEM_READ_ONLY, 	image_in_size_bytes);
    cl::Buffer buffer_outArrayY(context, CL_MEM_WRITE_ONLY, 	image_out_size_bytes);
    cl::Buffer buffer_outArrayX(context, CL_MEM_WRITE_ONLY, 	image_out_size_bytes);

    // Set kernel arguments:
    short hough_threshold = HOUGH_THRESHOLD;
    short hough_maxlines  = LINESMAX;

	krnl3.setArg(0, buffer_hough_inImage);
	krnl3.setArg(1, hough_threshold); // u should have a variable here !!
	krnl3.setArg(2, hough_maxlines);  // u should have a variable here !!
	krnl3.setArg(3, buffer_outArrayY);
	krnl3.setArg(4, buffer_outArrayX);
	krnl3.setArg(5, height);
	krnl3.setArg(6, width);

	// Initialize the buffers:
	cl::Event hg_event;
    printf("before enqueueWriteBuffer - houghlines");

    q.enqueueWriteBuffer(buffer_hough_inImage,      // buffer on the FPGA
                         CL_TRUE,             // blocking call
                         0,                   // buffer offset in bytes
						 image_in_size_bytes, // Size in bytes
						 out_img_edge.data,            // Pointer to the data to copy
                         nullptr, &hg_event);

	// Profiling Objects
	cl_ulong start_hough = 0;
	cl_ulong end_hough = 0;
	double diff_prof_hough = 0.0f;

	printf("before enqueueTask - hough transform");
	// Launch the kernel
	q.enqueueTask(krnl3, NULL, &hg_event);
	clWaitForEvents(1, (const cl_event*)&hg_event);

	printf("after enqueueTask - hough transform");

	hg_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_hough);
	hg_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_hough);
	diff_prof_hough = end_hough - start_hough;
	std::cout << "\n Profiling Hough Transform  " << (diff_prof_hough / 1000000) << "ms" << std::endl;

	// Copying Device result data to Host memory
	q.enqueueReadBuffer(buffer_outArrayY, // This buffers data will be read
            			CL_TRUE,          // blocking call
						0,                // offset
						image_out_size_bytes,
						outputrho.data(), // Data will be stored here
						nullptr, &hg_event);

	q.enqueueReadBuffer(buffer_outArrayX, // This buffers data will be read
            			CL_TRUE,          // blocking call
						0,                // offset
						image_out_size_bytes,
						outputtheta.data(), // Data will be stored here
						nullptr, &hg_event);

	q.finish();

	printf("after finish - hough transform");

    ///////////////////////////////// SW REFERENCE HOUGH TRANSFORM //////////////////////////////////////////////////
    // Running the reference function:
    cv::Mat crefxi, hlsxi;
    img_gray.copyTo(crefxi); // crefxi is a deep copy of img_gray
    img_gray.copyTo(hlsxi);
	std::vector<cv::Vec2f> linesxi;

/*
#define RHOSTEP 	     1
#define THETASTEP 	     2
#define HOUGH_THRESHOLD  75
#define MAX_LINES		 25

#define MINTHETA 0
#define MAXTHETA 180
*/

	float thetaval = (THETASTEP / 2.0);
	float angleref = (CV_PI * thetaval) / 180;

	xiHoughLinesstandard(ocv_img, linesxi, RHOSTEP, angleref, HOUGH_THRESHOLD, LINESMAX, MAXTHETA,
						 MINTHETA); // FLOATING POINT Reference code

	std::cout << "\n FLOATING POINT Reference Hough Transform code finished >>>> " << std::endl;


	FILE* fpre1 = fopen("hls_houghlines.txt", "w");
	FILE* fpre2 = fopen("ref_houghlines.txt", "w");

	for (size_t i = 0; i < linesxi.size(); i++) {
		fprintf(fpre1, "%f %f\n", outputrho[i], outputtheta[i]);
		fprintf(fpre2, "%f %f\n", linesxi[i][0], linesxi[i][1]);
		// Drawing the lines on the image.
	}

	fclose(fpre1);
	fclose(fpre2);


#if 1
	int heiby2 = (height/2);
	int wdtby2 = (width/2);

	// REFERENCE Drawing
	for( size_t i = 0; i < linesxi.size(); i++ )
	{
		float rho = linesxi[i][0], theta = linesxi[i][1];
		cv::Point pt1xi, pt2xi;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1xi.x = cvRound(x0 + 750*(-b) + (wdtby2));
		pt1xi.y = cvRound(y0 + 750*(a) + (heiby2));
		pt2xi.x = cvRound(x0 - 750*(-b) + (wdtby2));
		pt2xi.y = cvRound(y0 - 750*(a) + (heiby2));
		cv::line( crefxi, pt1xi, pt2xi, cv::Scalar(0,0,255), 1, CV_AA);
	}

	// HLS Drawing

	for( size_t i = 0; i < linesxi.size(); i++ )
	{
		float rho = outputrho[i], theta = outputtheta[i];
		cv::Point pt1xi, pt2xi;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1xi.x = cvRound(x0 + 1500*(-b) + (wdtby2)) ;
		pt1xi.y = cvRound(y0 + 1500*(a) + (heiby2));
		pt2xi.x = cvRound(x0 - 1500*(-b) + (wdtby2));
		pt2xi.y = cvRound(y0 - 1500*(a) + (heiby2));
		cv::line( hlsxi, pt1xi, pt2xi, cv::Scalar(0,0,255), 1, CV_AA);
	}


	// Write down the output images:
	cv::imwrite("hls_houghlines.png",hlsxi);
	cv::imwrite("ref_houghlines.png",crefxi);

#endif
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    /*			Destructors			*/
    in_img.~Mat();
    img_gray.~Mat();
    img_gray1.~Mat();
    hls_img.~Mat();
    ocv_img.~Mat();
    diff.~Mat();

    return 0;
}


NMSTYPE Filter3x3(
    NMSTYPE t0, NMSTYPE t1, NMSTYPE t2, NMSTYPE m0, NMSTYPE m1, NMSTYPE m2, NMSTYPE b0, NMSTYPE b1, NMSTYPE b2) {
    NMSTYPE value = false;
    int g0 = t0 + t2 + b0 + b2;
    int g1 = (t1 + b1 + m0 + m2) << 1;
    int g2 = m1 << 2;

    value = ((int)(g0 + g1 + g2) >> 4);
    return value;
}
void AverageGaussian(cv::Mat& src, cv::Mat& dst) {
    int i, j;
    NMSTYPE t0, t1, t2;
    NMSTYPE m0, m1, m2;
    NMSTYPE b0, b1, b2;
    NMSTYPE result;

    /*			First row			*/
    i = 0;
    for (j = 0; j < src.cols; j++) {
        if (j == 0) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = 0;
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if ((j > 0) && (j < src.cols - 1)) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = src.at<NMSTYPE>(i + 1, j + 1);
        } else if (j == src.cols - 1) {
            t0 = 0;
            t1 = 0;
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = 0;
            b0 = src.at<NMSTYPE>(i + 1, j - 1);
            b1 = src.at<NMSTYPE>(i + 1, j);
            b2 = 0;
        }
        result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result;
    }
    for (i = 1; i < src.rows - 1; i++) {
        for (j = 0; j < src.cols; j++) {
            if (j == 0) {
                t0 = 0;
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = 0;
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = 0;
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if ((j > 0) && (j < src.cols - 1)) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = src.at<NMSTYPE>(i - 1, j + 1);
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = src.at<NMSTYPE>(i, j + 1);
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = src.at<NMSTYPE>(i + 1, j + 1);
            } else if (j == src.cols - 1) {
                t0 = src.at<NMSTYPE>(i - 1, j - 1);
                t1 = src.at<NMSTYPE>(i - 1, j);
                t2 = 0;
                m0 = src.at<NMSTYPE>(i, j - 1);
                m1 = src.at<NMSTYPE>(i, j);
                m2 = 0;
                b0 = src.at<NMSTYPE>(i + 1, j - 1);
                b1 = src.at<NMSTYPE>(i + 1, j);
                b2 = 0;
            }
            result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
            dst.at<uchar>(i, j) = result;
        }
    }
    /*			Last row			*/
    i = src.rows - 1;
    for (j = 0; j < src.cols; j++) {
        if (j == 0) {
            t0 = 0;
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = src.at<NMSTYPE>(i - 1, j + 1);
            m0 = 0;
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = 0; // src.at<NMSTYPE>(i+1, j);
            b2 = 0; // src.at<NMSTYPE>(i+1, j+1);
        } else if ((j > 0) && (j < src.cols - 1)) {
            t0 = src.at<NMSTYPE>(i - 1, j - 1);
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = src.at<NMSTYPE>(i - 1, j + 1);
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = src.at<NMSTYPE>(i, j + 1);
            b0 = 0;
            b1 = 0;
            b2 = 0;
        } else if (j == src.cols - 1) {
            t0 = src.at<NMSTYPE>(i - 1, j - 1);
            t1 = src.at<NMSTYPE>(i - 1, j);
            t2 = 0;
            m0 = src.at<NMSTYPE>(i, j - 1);
            m1 = src.at<NMSTYPE>(i, j);
            m2 = 0;
            b0 = 0;
            b1 = 0;
            b2 = 0;
        }
        result = Filter3x3(t0, t1, t2, m0, m1, m2, b0, b1, b2);
        dst.at<uchar>(i, j) = result;
    }
}


float sinvalt[360] = {
    0.000000, 0.008727, 0.017452, 0.026177, 0.034899, 0.043619, 0.052336, 0.061049, 0.069756, 0.078459, 0.087156,
    0.095846, 0.104528, 0.113203, 0.121869, 0.130526, 0.139173, 0.147809, 0.156434, 0.165048, 0.173648, 0.182236,
    0.190809, 0.199368, 0.207912, 0.216440, 0.224951, 0.233445, 0.241922, 0.250380, 0.258819, 0.267238, 0.275637,
    0.284015, 0.292372, 0.300706, 0.309017, 0.317305, 0.325568, 0.333807, 0.342020, 0.350207, 0.358368, 0.366501,
    0.374607, 0.382684, 0.390731, 0.398749, 0.406737, 0.414693, 0.422618, 0.430511, 0.438371, 0.446198, 0.453991,
    0.461749, 0.469472, 0.477159, 0.484810, 0.492424, 0.499980, 0.507539, 0.515038, 0.522499, 0.529920, 0.537300,
    0.544639, 0.551937, 0.559193, 0.566407, 0.573577, 0.580703, 0.587786, 0.594823, 0.601815, 0.608762, 0.615662,
    0.622515, 0.629321, 0.636079, 0.642788, 0.649448, 0.656059, 0.662620, 0.669131, 0.675591, 0.681999, 0.688355,
    0.694659, 0.700910, 0.707107, 0.713251, 0.719340, 0.725375, 0.731354, 0.737278, 0.743145, 0.748956, 0.754710,
    0.760406, 0.766045, 0.771625, 0.777146, 0.782609, 0.788011, 0.793354, 0.798636, 0.803857, 0.809017, 0.814116,
    0.819152, 0.824127, 0.829038, 0.833886, 0.838671, 0.843392, 0.848048, 0.852641, 0.857168, 0.861629, 0.866026,
    0.870356, 0.874620, 0.878817, 0.882948, 0.887011, 0.891007, 0.894934, 0.898794, 0.902585, 0.906308, 0.909961,
    0.913545, 0.917060, 0.920505, 0.923879, 0.927184, 0.930417, 0.933580, 0.936672, 0.939692, 0.942641, 0.945518,
    0.948323, 0.951056, 0.953717, 0.956305, 0.958820, 0.961261, 0.963630, 0.965926, 0.968147, 0.970295, 0.972370,
    0.974370, 0.976296, 0.978147, 0.979924, 0.981627, 0.983255, 0.984807, 0.986285, 0.987688, 0.989016, 0.990268,
    0.991445, 0.992546, 0.993572, 0.994522, 0.995396, 0.996195, 0.996917, 0.997564, 0.998135, 0.998629, 0.999048,
    0.999391, 0.999657, 0.999848, 0.999962, 0.999980, 0.999962, 0.999848, 0.999657, 0.999391, 0.999048, 0.998629,
    0.998135, 0.997564, 0.996917, 0.996195, 0.995396, 0.994522, 0.993572, 0.992546, 0.991445, 0.990268, 0.989016,
    0.987688, 0.986285, 0.984807, 0.983255, 0.981627, 0.979924, 0.978147, 0.976296, 0.97437,  0.97237,  0.970295,
    0.968147, 0.965926, 0.96363,  0.961261, 0.95882,  0.956305, 0.953717, 0.951056, 0.948323, 0.945518, 0.942641,
    0.939692, 0.936672, 0.93358,  0.930417, 0.927184, 0.923879, 0.920505, 0.91706,  0.913545, 0.909961, 0.906308,
    0.902585, 0.898794, 0.894934, 0.891007, 0.887011, 0.882948, 0.878817, 0.87462,  0.870356, 0.866026, 0.861629,
    0.857168, 0.852641, 0.848048, 0.843392, 0.838671, 0.833886, 0.829038, 0.824127, 0.819152, 0.814116, 0.809017,
    0.803857, 0.798636, 0.793354, 0.788011, 0.782609, 0.777146, 0.771625, 0.766045, 0.760406, 0.75471,  0.748956,
    0.743145, 0.737278, 0.731354, 0.725375, 0.71934,  0.713251, 0.707107, 0.70091,  0.694659, 0.688355, 0.681999,
    0.675591, 0.669131, 0.66262,  0.656059, 0.649448, 0.642788, 0.636079, 0.629321, 0.622515, 0.615662, 0.608762,
    0.601815, 0.594823, 0.587786, 0.580703, 0.573577, 0.566407, 0.559193, 0.551937, 0.544639, 0.5373,   0.52992,
    0.522499, 0.515038, 0.507539, 0.5,      0.492424, 0.48481,  0.477159, 0.469472, 0.461749, 0.453991, 0.446198,
    0.438371, 0.430511, 0.422618, 0.414693, 0.406737, 0.398749, 0.390731, 0.382684, 0.374607, 0.366501, 0.358368,
    0.350207, 0.34202,  0.333807, 0.325568, 0.317305, 0.309017, 0.300706, 0.292372, 0.284015, 0.275637, 0.267238,
    0.258819, 0.25038,  0.241922, 0.233445, 0.224951, 0.21644,  0.207912, 0.199368, 0.190809, 0.182236, 0.173648,
    0.165048, 0.156434, 0.147809, 0.139173, 0.130526, 0.121869, 0.113203, 0.104528, 0.095846, 0.087156, 0.078459,
    0.069756, 0.061049, 0.052336, 0.043619, 0.034899, 0.026177, 0.017452, 0.008727};
float cosvalt[360] = {
    0.999980,  0.999962,  0.999848,  0.999657,  0.999391,  0.999048,  0.998629,  0.998135,  0.997564,  0.996917,
    0.996195,  0.995396,  0.994522,  0.993572,  0.992546,  0.991445,  0.990268,  0.989016,  0.987688,  0.986285,
    0.984807,  0.983255,  0.981627,  0.979924,  0.978147,  0.976296,  0.97437,   0.97237,   0.970295,  0.968147,
    0.965926,  0.96363,   0.961261,  0.95882,   0.956305,  0.953717,  0.951056,  0.948323,  0.945518,  0.942641,
    0.939692,  0.936672,  0.93358,   0.930417,  0.927184,  0.923879,  0.920505,  0.91706,   0.913545,  0.909961,
    0.906308,  0.902585,  0.898794,  0.894934,  0.891007,  0.887011,  0.882948,  0.878817,  0.87462,   0.870356,
    0.866026,  0.861629,  0.857168,  0.852641,  0.848048,  0.843392,  0.838671,  0.833886,  0.829038,  0.824127,
    0.819152,  0.814116,  0.809017,  0.803857,  0.798636,  0.793354,  0.788011,  0.782609,  0.777146,  0.771625,
    0.766045,  0.760406,  0.75471,   0.748956,  0.743145,  0.737278,  0.731354,  0.725375,  0.71934,   0.713251,
    0.707107,  0.70091,   0.694659,  0.688355,  0.681999,  0.675591,  0.669131,  0.66262,   0.656059,  0.649448,
    0.642788,  0.636079,  0.629321,  0.622515,  0.615662,  0.608762,  0.601815,  0.594823,  0.587786,  0.580703,
    0.573577,  0.566407,  0.559193,  0.551937,  0.544639,  0.5373,    0.52992,   0.522499,  0.515038,  0.507539,
    0.5,       0.492424,  0.48481,   0.477159,  0.469472,  0.461749,  0.453991,  0.446198,  0.438371,  0.430511,
    0.422618,  0.414693,  0.406737,  0.398749,  0.390731,  0.382684,  0.374607,  0.366501,  0.358368,  0.350207,
    0.34202,   0.333807,  0.325568,  0.317305,  0.309017,  0.300706,  0.292372,  0.284015,  0.275637,  0.267238,
    0.258819,  0.25038,   0.241922,  0.233445,  0.224951,  0.21644,   0.207912,  0.199368,  0.190809,  0.182236,
    0.173648,  0.165048,  0.156434,  0.147809,  0.139173,  0.130526,  0.121869,  0.113203,  0.104528,  0.095846,
    0.087156,  0.078459,  0.069756,  0.061049,  0.052336,  0.043619,  0.034899,  0.026177,  0.017452,  0.008727,
    0.000000,  -0.008727, -0.017452, -0.026177, -0.034899, -0.043619, -0.052336, -0.061049, -0.069756, -0.078459,
    -0.087156, -0.095846, -0.104528, -0.113203, -0.121869, -0.130526, -0.139173, -0.147809, -0.156434, -0.165048,
    -0.173648, -0.182236, -0.190809, -0.199368, -0.207912, -0.216440, -0.224951, -0.233445, -0.241922, -0.250380,
    -0.258819, -0.267238, -0.275637, -0.284015, -0.292372, -0.300706, -0.309017, -0.317305, -0.325568, -0.333807,
    -0.342020, -0.350207, -0.358368, -0.366501, -0.374607, -0.382684, -0.390731, -0.398749, -0.406737, -0.414693,
    -0.422618, -0.430511, -0.438371, -0.446198, -0.453991, -0.461749, -0.469472, -0.477159, -0.484810, -0.492424,
    -0.499980, -0.507539, -0.515038, -0.522499, -0.529920, -0.537300, -0.544639, -0.551937, -0.559193, -0.566407,
    -0.573577, -0.580703, -0.587786, -0.594823, -0.601815, -0.608762, -0.615662, -0.622515, -0.629321, -0.636079,
    -0.642788, -0.649448, -0.656059, -0.662620, -0.669131, -0.675591, -0.681999, -0.688355, -0.694659, -0.700910,
    -0.707107, -0.713251, -0.719340, -0.725375, -0.731354, -0.737278, -0.743145, -0.748956, -0.754710, -0.760406,
    -0.766045, -0.771625, -0.777146, -0.782609, -0.788011, -0.793354, -0.798636, -0.803857, -0.809017, -0.814116,
    -0.819152, -0.824127, -0.829038, -0.833886, -0.838671, -0.843392, -0.848048, -0.852641, -0.857168, -0.861629,
    -0.866026, -0.870356, -0.874620, -0.878817, -0.882948, -0.887011, -0.891007, -0.894934, -0.898794, -0.902585,
    -0.906308, -0.909961, -0.913545, -0.917060, -0.920505, -0.923879, -0.927184, -0.930417, -0.933580, -0.936672,
    -0.939692, -0.942641, -0.945518, -0.948323, -0.951056, -0.953717, -0.956305, -0.958820, -0.961261, -0.963630,
    -0.965926, -0.968147, -0.970295, -0.972370, -0.974370, -0.976296, -0.978147, -0.979924, -0.981627, -0.983255,
    -0.984807, -0.986285, -0.987688, -0.989016, -0.990268, -0.991445, -0.992546, -0.993572, -0.994522, -0.995396,
    -0.996195, -0.996917, -0.997564, -0.998135, -0.998629, -0.999048, -0.999391, -0.999657, -0.999848, -0.999962};


struct LinePolar {
    float rho;
    float angle;
};

struct hough_cmp_gt {
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const { return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2); }
    const int* aux;
};


void xiHoughLinesstandard(cv::Mat& img,
                          std::vector<cv::Vec2f>& lines,
                          float rho,
                          float theta,
                          int threshold,
                          int linesMax,
                          int maxtheta,
                          int mintheta) {
    int i, j;
    float irho = 1 / rho;

    CV_Assert(img.type() == CV_8UC1);

    const uchar* image = img.ptr();
    int step = (int)img.step;
    int width = img.cols;
    int height = img.rows;
    double max_theta;
    double min_theta;

    if (maxtheta > 0) max_theta = (CV_PI * maxtheta) / 180;
    if (mintheta > 0) min_theta = (CV_PI * mintheta) / 180;

    if (max_theta < min_theta) {
        CV_Error(CV_StsBadArg, "max_theta must be greater than min_theta");
    }
    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound((sqrt(width * width + height * height)) / rho);

    cv::AutoBuffer<int> _accum((numangle + 2) * (numrho + 2));
    std::vector<int> _sort_buf;
    cv::AutoBuffer<float> _tabSin(numangle);
    cv::AutoBuffer<float> _tabCos(numangle);
    int* accum = _accum;
    float *tabSin = _tabSin, *tabCos = _tabCos;

    memset(accum, 0, sizeof(accum[0]) * (numangle + 2) * (numrho + 2));

    float thetaind = (theta * 180) / CV_PI;

    int ang = 2 * mintheta;

    for (int n = 0; n < numangle; ang += (2 * thetaind), n++) {
        tabSin[n] = (float)sinvalt[ang] * irho;
        tabCos[n] = (float)cosvalt[ang] * irho;
    }

    float temp[360], tempsinval[360], tempcosval[360];
    for (int i = 0; i < 360; i++) {
        temp[i] = 0.0;
        tempsinval[i] = 0.0;
        tempcosval[i] = 0.0;
    }
    // stage 1. fill accumulator
    float r1;

    int hei = (height / 2);
    int wdt = (width / 2);

    for (int ki = 0; ki < numangle; ki++) {
        tempsinval[ki] = (-hei) * tabSin[ki];
        tempcosval[ki] = (-wdt) * tabCos[ki];
    }

    for (i = 0; i < height; i++) // i -->row
    {
        for (int ki = 0; ki < numangle; ki++) {
            if (i > 0) tempsinval[ki] = tempsinval[ki] + tabSin[ki];
        }

        for (j = 0; j < width; j++) // j-->col
        {
            for (int n = 0; n < numangle; n++) {
                int rho1;

                if (j == 0) {
                    r1 = tempcosval[n] + tempsinval[n];
                } else {
                    r1 = temp[n] + tabCos[n];
                }

                temp[n] = r1;

                rho1 = cvRound(r1) + ((numrho) / 2);

                if (image[(i) * (step) + (j)] != 0) {
                    accum[(n + 1) * (numrho + 2) + rho1 + 1]++;
                }
            }
        }
    }

    // stage 2. find local maximums
    for (int r = 0; r < numrho; r++)
        for (int n = 0; n < numangle; n++) {
            int base = (n + 1) * (numrho + 2) + r + 1;
            if (accum[base] > threshold && accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2])
                _sort_buf.push_back(base);
        }

    // stage 3. sort the detected lines by accumulator value
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    double scale = 1. / (numrho + 2);

    for (i = 0; i < linesMax; i++) {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = cvFloor(idx * scale) - 1;
        int r = idx - (n + 1) * (numrho + 2) - 1;
        line.rho = (r - (numrho)*0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(cv::Vec2f(line.rho, line.angle));
    }
}

