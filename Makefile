CXX ?= g++
AR ?= ar

BUILD_DIR := build
INCLUDE_DIR := include
SOURCE_DIR := src
APP_DIR := apps

CORE_NAMES := camera_resolver frame_synchronizer panorama_composer \
              panorama_pipeline warp_producer gpu_warp_roi camer_pip \
              mpp v4l2_streaming dma_alloc
CORE_OBJECTS := $(addprefix $(BUILD_DIR)/,$(addsuffix .o,$(CORE_NAMES)))
DISPLAY_OBJECT := $(BUILD_DIR)/panorama_display.o
LEASE_TEST_OBJECT := $(BUILD_DIR)/panorama_output_lease_test.o
DEPENDENCY_FILES := $(CORE_OBJECTS:.o=.d) $(DISPLAY_OBJECT:.o=.d) \
                    $(LEASE_TEST_OBJECT:.o=.d)

LIBRARY := $(BUILD_DIR)/libpanorama_pipeline.a
DISPLAY := $(BUILD_DIR)/panorama_display
LEASE_TEST := $(BUILD_DIR)/panorama_output_lease_test

CPPFLAGS := -I$(INCLUDE_DIR) -I/usr/local/include
CXXFLAGS := -O2 -g -std=c++17 -Wall -Wextra -pthread -MMD -MP
DISPLAY_CPPFLAGS := $(CPPFLAGS) -I/usr/include/opencv4
LDFLAGS := -L/usr/lib/aarch64-linux-gnu/libmali-x11 \
           -L/usr/local/lib \
           -Wl,-rpath,/usr/lib/aarch64-linux-gnu/libmali-x11 \
           -Wl,-rpath,/usr/local/lib
CORE_LIBS := -l:libOpenCL.so.1 -lrga -lrockchip_mpp -pthread
DISPLAY_LIBS := -lopencv_highgui -lopencv_core

.PHONY: all clean check-libs

all: $(LIBRARY) $(DISPLAY) $(LEASE_TEST)

$(LIBRARY): $(CORE_OBJECTS)
	$(AR) rcs $@ $^

$(DISPLAY): $(DISPLAY_OBJECT) $(LIBRARY)
	$(CXX) $(DISPLAY_OBJECT) $(LIBRARY) $(LDFLAGS) \
	      $(CORE_LIBS) $(DISPLAY_LIBS) -o $@

$(LEASE_TEST): $(LEASE_TEST_OBJECT) $(LIBRARY)
	$(CXX) $(LEASE_TEST_OBJECT) $(LIBRARY) $(LDFLAGS) \
	      $(CORE_LIBS) -o $@

$(BUILD_DIR)/panorama_display.o: $(APP_DIR)/panorama_display.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(DISPLAY_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/panorama_output_lease_test.o: $(APP_DIR)/panorama_output_lease_test.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

check-libs: $(DISPLAY)
	@ldd $(DISPLAY) | grep -E 'libmali|rockchip_mpp|librga|opencv'
	@ldd $(DISPLAY) | grep -q '/usr/lib/aarch64-linux-gnu/libmali-x11/libmali.so.1' || \
	    { echo 'ERROR: Mali OpenCL driver is not selected'; exit 1; }
	@echo 'Runtime library check: OK'

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPENDENCY_FILES)
