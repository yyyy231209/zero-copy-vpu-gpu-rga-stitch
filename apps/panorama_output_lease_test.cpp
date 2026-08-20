#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#include "panorama_pipeline.h"

namespace {

std::uint64_t hash_visible(const PanoramaFrameRef &frame)
{
    const auto *base = static_cast<const std::uint8_t *>(frame.data());
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (int y = 0; y < frame.height(); ++y) {
        const auto *row = base + static_cast<std::size_t>(y) *
                                    frame.stride() * 3U;
        for (std::size_t x = 0;
             x < static_cast<std::size_t>(frame.width()) * 3U; ++x) {
            hash ^= row[x];
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

bool contract_ok(const PanoramaFrameRef &frame)
{
    return frame.valid() && frame.dma_fd() >= 0 && frame.data() != nullptr &&
           frame.bytes() == static_cast<std::size_t>(2256) * 330U * 3U &&
           frame.width() == 2248 && frame.height() == 330 &&
           frame.stride() == 2256 && frame.release_callback() != nullptr &&
           frame.release_context() != nullptr;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s assets_dir report.txt\n", argv[0]);
        return 2;
    }

    PanoramaPipeline pipeline;
    if (!pipeline.init(argv[1]) || !pipeline.start())
        return 3;

    std::array<PanoramaFrameRef, 6> held;
    std::array<std::uint64_t, 6> hashes{};
    std::set<int> fds;
    bool contracts = true;
    bool sequences = true;
    for (std::size_t i = 0; i < held.size(); ++i) {
        if (!pipeline.acquire(&held[i], 1500)) {
            pipeline.stop();
            return 4;
        }
        contracts &= contract_ok(held[i]);
        if (i > 0)
            sequences &= held[i].sequence() > held[i - 1].sequence();
        hashes[i] = hash_visible(held[i]);
        contracts &= hashes[i] != 0;
        fds.insert(held[i].dma_fd());
    }

    PanoramaFrameRef seventh;
    const bool seventh_blocked = !pipeline.acquire(&seventh, 100);

    const int released_fd = held[2].dma_fd();
    held[2].release();
    held[2].release();  // A detached reference must be an idempotent no-op.
    PanoramaFrameRef replacement;
    const bool replacement_acquired = pipeline.acquire(&replacement, 1500);
    const bool released_slot_reused = replacement_acquired &&
                                      replacement.dma_fd() == released_fd;
    contracts &= !replacement_acquired || contract_ok(replacement);

    PanoramaFrameRef moved = std::move(replacement);
    const bool move_ok = !replacement.valid() && moved.valid();
    moved.release();
    for (auto &frame : held)
        frame.release();

    PanoramaFrame legacy_first;
    PanoramaFrame legacy_second;
    const bool legacy_first_ok = pipeline.read(&legacy_first, 1500) &&
                                 legacy_first.dma_fd >= 0 &&
                                 legacy_first.data != nullptr &&
                                 legacy_first.width == 2248 &&
                                 legacy_first.height == 330 &&
                                 legacy_first.stride == 2256;
    const bool legacy_second_ok = pipeline.read(&legacy_second, 1500) &&
                                  legacy_second.sequence >
                                      legacy_first.sequence;
    const bool legacy_read_ok = legacy_first_ok && legacy_second_ok;

    pipeline.stop();
    pipeline.close();
    const PanoramaPipelineStats stats = pipeline.stats();
    const bool passed = contracts && sequences && fds.size() == held.size() &&
                        seventh_blocked && released_slot_reused && move_ok &&
                        legacy_read_ok &&
                        stats.output_wait_timeouts >= 1 &&
                        stats.invalid_output_releases == 0 &&
                        stats.outstanding_output_leases == 0 &&
                        stats.errors == 0 && !pipeline.fatal();

    const std::filesystem::path report_path(argv[2]);
    if (!report_path.parent_path().empty())
        std::filesystem::create_directories(report_path.parent_path());
    std::ofstream report(report_path);
    report << "passed=" << (passed ? 1 : 0) << '\n'
           << "unique_fds=" << fds.size() << '\n'
           << "contracts=" << (contracts ? 1 : 0) << '\n'
           << "sequences=" << (sequences ? 1 : 0) << '\n'
           << "seventh_blocked=" << (seventh_blocked ? 1 : 0) << '\n'
           << "released_slot_reused=" << (released_slot_reused ? 1 : 0) << '\n'
           << "move_ok=" << (move_ok ? 1 : 0) << '\n'
           << "legacy_read_ok=" << (legacy_read_ok ? 1 : 0) << '\n'
           << "output_wait_timeouts=" << stats.output_wait_timeouts << '\n'
           << "invalid_output_releases=" << stats.invalid_output_releases << '\n'
           << "outstanding_output_leases=" << stats.outstanding_output_leases << '\n'
           << "pipeline_errors=" << stats.errors << '\n';
    for (std::size_t i = 0; i < hashes.size(); ++i)
        report << "frame" << i << "_hash=" << std::hex << hashes[i]
               << std::dec << '\n';

    std::printf("PANORAMA_OUTPUT_LEASE_TEST_%s unique_fds=%zu blocked=%d "
                "reused=%d invalid=%llu outstanding=%llu\n",
                passed ? "PASSED" : "FAILED", fds.size(), seventh_blocked,
                released_slot_reused,
                static_cast<unsigned long long>(stats.invalid_output_releases),
                static_cast<unsigned long long>(stats.outstanding_output_leases));
    return passed ? 0 : 5;
}
