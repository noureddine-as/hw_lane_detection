set substitute-path '/home/centos/labs/lane_detector_v2/Emulation-SW/krnl_canny.build/link/int/xo/canny_accel/canny_accel/cpu_sources/' '/home/centos/labs/lane_detector_v2/src/'
set substitute-path '/home/centos/labs/lane_detector_v2/Emulation-SW/krnl_canny.build/link/int/xo/edgetracing_accel/edgetracing_accel/cpu_sources/' '/home/centos/labs/lane_detector_v2/src/'
set substitute-path '/home/centos/labs/lane_detector_v2/Emulation-SW/krnl_canny.build/link/int/xo/houghlines_accel/houghlines_accel/cpu_sources/' '/home/centos/labs/lane_detector_v2/src/'
source /opt/xilinx/xrt/share/appdebug/appdebug.py
handle SIGUSR1 nostop pass
handle SIGTERM nostop pass
