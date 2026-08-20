#include "camera_resolver.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <vector>

namespace {

bool supports_mjpeg_size(int fd, int width, int height)
{
    v4l2_fmtdesc format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    for (format.index = 0; ioctl(fd, VIDIOC_ENUM_FMT, &format) == 0;
         ++format.index) {
        if (format.pixelformat != V4L2_PIX_FMT_MJPEG)
            continue;
        v4l2_frmsizeenum size{};
        size.pixel_format = V4L2_PIX_FMT_MJPEG;
        for (size.index = 0; ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0;
             ++size.index) {
            if (size.type == V4L2_FRMSIZE_TYPE_DISCRETE &&
                static_cast<int>(size.discrete.width) == width &&
                static_cast<int>(size.discrete.height) == height)
                return true;
            if (size.type == V4L2_FRMSIZE_TYPE_STEPWISE &&
                width >= static_cast<int>(size.stepwise.min_width) &&
                width <= static_cast<int>(size.stepwise.max_width) &&
                height >= static_cast<int>(size.stepwise.min_height) &&
                height <= static_cast<int>(size.stepwise.max_height))
                return true;
        }
    }
    return false;
}

}  // namespace

bool resolve_usb_camera_by_hub_port(int downstream_port,
                                    int required_width,
                                    int required_height,
                                    std::string *device_path,
                                    std::string *diagnostic)
{
    if (!device_path || downstream_port <= 0 ||
        required_width <= 0 || required_height <= 0)
        return false;

    const std::string token = "1-1.4." + std::to_string(downstream_port);
    std::vector<std::string> matches;
    std::ostringstream details;
    std::error_code error;
    const std::filesystem::path root("/sys/class/video4linux");
    for (const auto &entry : std::filesystem::directory_iterator(root, error)) {
        if (error)
            break;
        const std::string name = entry.path().filename().string();
        if (name.rfind("video", 0) != 0)
            continue;

        const std::filesystem::path resolved =
            std::filesystem::canonical(entry.path() / "device", error);
        if (error) {
            error.clear();
            continue;
        }
        if (resolved.string().find(token) == std::string::npos)
            continue;

        const std::string candidate = "/dev/" + name;
        const int fd = open(candidate.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) {
            details << candidate << ": open failed: " << std::strerror(errno) << "; ";
            continue;
        }
        v4l2_capability capability{};
        bool usable = ioctl(fd, VIDIOC_QUERYCAP, &capability) == 0;
        const uint32_t caps =
            (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                ? capability.device_caps
                : capability.capabilities;
        usable = usable && (caps & V4L2_CAP_VIDEO_CAPTURE) &&
                 (caps & V4L2_CAP_STREAMING) &&
                 supports_mjpeg_size(fd, required_width, required_height);
        close(fd);
        details << candidate << (usable ? ": image match; " : ": rejected; ");
        if (usable)
            matches.push_back(candidate);
    }

    if (diagnostic)
        *diagnostic = details.str();
    if (matches.size() != 1)
        return false;
    *device_path = matches.front();
    return true;
}
