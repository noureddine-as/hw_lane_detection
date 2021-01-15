#
# this file was created by a computer. trust it.
#

# compiler tools
XILINX_VITIS ?= /opt/Xilinx/Vitis/2020.1
XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VIVADO ?= /opt/Xilinx/Vivado/2020.1
XILINX_VIVADO_HLS ?= $(XILINX_VITIS)/Vivado_HLS

HOST_CXX ?= g++
VPP ?= ${XILINX_VITIS}/bin/v++
EMCONFIGUTIL = ${XILINX_VITIS}/bin/emconfigutil --od .
RM = rm -f
RMDIR = rm -rf

VITIS_PLATFORM = xilinx_aws-vu9p-f1_shell-v04261818_201920_2
VITIS_PLATFORM_DIR = /home/centos/src/project_data/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_2
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/xilinx_aws-vu9p-f1_shell-v04261818_201920_2.xpfm

# host compiler global settings
CXXFLAGS += -std=c++0x -D__SDSVHLS__ -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I/home/centos/labs/lane_detector_v2/libs/xf_opencv/ext/xcl2 -I/home/centos/labs/lane_detector_v2/libs/xf_opencv/L1/include -I/usr/include/opencv -I/home/centos/labs/lane_detector_v2/src/build -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -O0 -g -Wall -O3 -c -fmessage-length=0
LDFLAGS += -lxml2 -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_video -lopencv_calib3d -lopencv_highgui -lxilinxopencl -lpthread -lrt -lstdc++ -lmpfr -lgmp -lhlsmc++-GCC46 -lIp_floating_point_v7_0_bitacc_cmodel -lIp_xfft_v9_1_bitacc_cmodel -lIp_fir_compiler_v7_2_bitacc_cmodel -lIp_dds_compiler_v6_0_bitacc_cmodel -L$(XILINX_XRT)/lib/ -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -L$(XILINX_VIVADO)/lnx64/lib/csim -L$(XILINX_VIVADO)/lnx64/tools/dds_v6_0 -L$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -L$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath-link,$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_VIVADO)/lnx64/lib/csim -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/dds_v6_0

# hardware compiler shared settings
VPP_OPTS = --target sw_emu

#
# OpenCL kernel files
#

BINARY_CONTAINERS += krnl_canny.xclbin

BUILD_SUBDIRS += krnl_canny.build
KRNL_CANNY_OBJS += krnl_canny.build/canny_accel.xo
ALL_KERNEL_OBJS += krnl_canny.build/canny_accel.xo
BUILD_SUBDIRS += krnl_canny.build
KRNL_CANNY_OBJS += krnl_canny.build/edgetracing_accel.xo
ALL_KERNEL_OBJS += krnl_canny.build/edgetracing_accel.xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

#
# host files
#

HOST_OBJECTS += libs/xf_opencv/ext/xcl2/xcl2.o
HOST_OBJECTS += src/xf_canny_tb.o
HOST_OBJECTS += src/xf_houghlines_accel.o

HOST_EXE = lane_detector_v2

BUILD_SUBDIRS += libs/xf_opencv/ext/xcl2/
BUILD_SUBDIRS += src/

EMCONFIG_FILE = emconfig.json

#
# primary build targets
#

.PHONY: all clean
all: $(BINARY_CONTAINERS) $(HOST_EXE) $(EMCONFIG_FILE)

.NOTPARALLEL: clean

clean-host:
	-$(RM) $(HOST_EXE) $(HOST_OBJECTS)

clean-accelerators:
	-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES)  $(EMCONFIG_FILE)
	-$(RM) *.xclbin.sh *.xclbin.info *.xclbin.link_summary*
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil

clean-package:
	-${RMDIR} package
	-${RMDIR} package.build

clean: clean-host clean-accelerators clean-package

.PHONY: incremental
incremental: all

include makeemconfig.mk

nothing:

#
# binary container: krnl_canny.xclbin
#

krnl_canny.build/canny_accel.xo: ../src/xf_canny_accel.cpp ../src/xf_canny_config.h krnl_canny-canny_accel-compile.cfg common-config.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config common-config.cfg --config krnl_canny-canny_accel-compile.cfg -I/home/centos/labs/lane_detector_v2/libs/xf_opencv/L1/include -I/home/centos/labs/lane_detector_v2/src/build -o"$@" "$<"

krnl_canny.build/edgetracing_accel.xo: ../src/xf_edgetracing_accel.cpp ../src/xf_canny_config.h krnl_canny-edgetracing_accel-compile.cfg common-config.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" -D__SDA_MEM_MAP__ --config common-config.cfg --config krnl_canny-edgetracing_accel-compile.cfg -I/home/centos/labs/lane_detector_v2/libs/xf_opencv/L1/include -I/home/centos/labs/lane_detector_v2/src/build -o"$@" "$<"

krnl_canny.xclbin: $(KRNL_CANNY_OBJS) krnl_canny-link.cfg common-config.cfg
	-@echo $(VPP) $(VPP_OPTS) --link --config common-config.cfg --config krnl_canny-link.cfg -o"$@" $(KRNL_CANNY_OBJS) > krnl_canny.xclbin.sh
	$(VPP) $(VPP_OPTS) --link --config common-config.cfg --config krnl_canny-link.cfg -o"$@" $(KRNL_CANNY_OBJS)

#
# host rules
#

libs/xf_opencv/ext/xcl2/xcl2.o: ../libs/xf_opencv/ext/xcl2/xcl2.cpp ../libs/xf_opencv/ext/xcl2/xcl2.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/xf_canny_tb.o: ../src/xf_canny_tb.cpp ../libs/xf_opencv/L1/include/common/xf_headers.hpp ../src/xf_canny_config.h ../src/xf_houghlines_config.h ../libs/xf_opencv/ext/xcl2/xcl2.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/xf_houghlines_accel.o: ../src/xf_houghlines_accel.cpp ../src/xf_houghlines_config.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
	-@echo $(VPP) --package --config package.cfg  krnl_canny.xclbin
	-@$(VPP) --package --config package.cfg  krnl_canny.xclbin


#
# emulation configuration file
#

$(EMCONFIG_FILE): makeemconfig.mk
	$(EMCONFIGUTIL) --nd $(NUMBER_OF_DEVICES)  --platform $(VITIS_PLATFORM_PATH)
	-@$(RMDIR) TempConfig .Xil/configutil*

