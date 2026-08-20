#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>

#include <opencv2/highgui.hpp>

#include "panorama_pipeline.h"

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s assets_dir\n", argv[0]);
        return 2;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    PanoramaPipeline pipeline;
    if (!pipeline.init(argv[1]) || !pipeline.start())
        return 1;

    cv::namedWindow("RK3588 Panorama", cv::WINDOW_AUTOSIZE);
    uint64_t report_frames = 0;
    auto report_start = std::chrono::steady_clock::now();

    while (!g_stop.load()) {
        PanoramaFrame frame;
        if (!pipeline.read(&frame)) {
            if (pipeline.fatal())
                break;
            continue;
        }

        cv::Mat image(frame.height, frame.width, CV_8UC3, frame.data,
                      static_cast<std::size_t>(frame.stride) * 3);
        cv::imshow("RK3588 Panorama", image);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q')
            break;

        ++report_frames;
        const auto now = std::chrono::steady_clock::now();
        const double seconds =
            std::chrono::duration<double>(now - report_start).count();
        if (seconds >= 2.0) {
            std::printf("FPS %.2f, camera spread %.2f ms\n",
                        report_frames / seconds,
                        frame.camera_spread_ns / 1e6);
            std::fflush(stdout);
            report_frames = 0;
            report_start = now;
        }
    }

    pipeline.stop();
    const auto stats = pipeline.stats();
    cv::destroyAllWindows();
    std::printf(
        "DONE frames=%llu errors=%llu RGA=%.3fms GPU=%.3fms BGR=%.3fms\n",
        static_cast<unsigned long long>(stats.frames),
        static_cast<unsigned long long>(stats.errors),
        stats.rga_body_average_ms,
        stats.gpu_seam_average_ms,
        stats.bgr_convert_average_ms);
    return stats.frames > 0 && stats.errors == 0 ? 0 : 1;
}
