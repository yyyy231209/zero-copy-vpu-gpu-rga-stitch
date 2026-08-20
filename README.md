# RK3588 Four-Camera Real-Time Panorama Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-RK3588-red.svg)](https://www.rock-chips.com)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)]()
[![CI](https://github.com/yyyy231209/zero-copy-vpu-gpu-rga-stitch/actions/workflows/ci.yml/badge.svg)](https://github.com/yyyy231209/zero-copy-vpu-gpu-rga-stitch/actions/workflows/ci.yml)
[![Wiki](https://img.shields.io/badge/Wiki-文档-2ea44f.svg)](https://github.com/yyyy231209/zero-copy-vpu-gpu-rga-stitch/wiki)

RK3588 平台零拷贝四路摄像头实时全景拼接管线：V4L2 采集 → MPP 硬解码 → Mali GPU OpenCL 几何变换 → RGA 主体拷贝 + GPU 窄接缝融合 → DMA-BUF 全景输出。

A zero-copy four-camera real-time panorama stitching pipeline on Rockchip RK3588: V4L2 capture → MPP hardware decode → Mali GPU OpenCL warp → RGA body copy + GPU narrow-seam blending → DMA-BUF panorama output.

---

## ✨ 技术亮点 / Highlights

| 亮点 | 说明 |
|---|---|
| 🚀 **零拷贝** | 全程 DMA-BUF fd 传递，无 CPU 图像拷贝；输出 BGR DMA-BUF 可直接被下游 NPU 绑定 |
| ⚡ **高吞吐实时** | 帧率瓶颈在输入源而非管线：30 FPS 摄像头跑满 30 FPS，60 FPS 摄像头可达 ~50 FPS，120 FPS 摄像头可达 ~110 FPS |
| 🎯 **混合异构加速** | **VPU(MPP) 硬解码** + **Mali GPU OpenCL** remap（`clImportMemoryARM` 导入 DMA-BUF）+ RGA 主体拷贝 + GPU 接缝融合 |
| 🔧 **自编译官方 MPP** | 系统自带 `librockchip_mpp.so` 存在 MJPEG 解码 segfault bug，**从官方源码重新编译 MPP 并安装到 `/usr/local`**，不依赖系统库 |
| 🧵 **帧同步** | `FrameSynchronizer` 按时间戳配对四路，超时丢旧帧不积压 |
| 📦 **多级缓冲池** | 每路 WarpProducer 3 槽（`FREE→WRITING→READY→READING`）+ 全景 NV12 3 槽轮转 + BGR 输出 6 槽租赁池（`Free→Writing→Leased`） |
| 🔧 **无 OpenCV 依赖** | 生产库 `libpanorama_pipeline.a` 不依赖 OpenCV（仅显示程序使用） |

## 🏗️ 系统架构 / Architecture

### 数据流图 / Data Flow

```text
┌─────────────────────────────────────────────────────────────┐
│            Rockchip RK3588 SoC                              │
│                                                             │
│  CAM1 ─┐                                                    │
│  CAM2 ─┼─► V4L2 MJPG ─► MPP 硬解码(NV12) ─┐                │
│  CAM3 ─┤            (DMA-BUF fd)           │                │
│  CAM4 ─┘                                    ▼               │
│                              GPU OpenCL remap (预标定 LUT)  │
│                                    │                        │
│                                    ▼                        │
│                     RGA 主体拷贝(95.7%) + GPU 窄接缝融合     │
│                                    │                        │
│                                    ▼                        │
│                      Panorama NV12 DMA-BUF (2390×720)       │
│                                    │                        │
│                                    ▼                        │
│                  RGA 1:1 裁剪+转换 BGR888 (2248×330)        │
│                                    │                        │
│                                    ▼                        │
│                    PanoramaFrame::dma_fd (输出契约)          │
└─────────────────────────────────────────────────────────────┘
```

### 软件模块架构 / Software Architecture

```text
                    ┌─────────────────────────────────────┐
                    │    PanoramaPipeline (门面 API)      │
                    │  init() / start() / acquire() /     │
                    │  read() / stop() / stats()          │
                    └──────────────┬──────────────────────┘
                                   │ 创建并驱动
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌───────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ WarpProducer  │   │ FrameSynchronizer    │   │  PanoramaComposer    │
│   ×4 线程     │   │ 按时间戳配对四路       │   │  合成器               │
│               │   │ 超时丢旧帧不积压       │   │                      │
│ ┌───────────┐ │   └──────────┬───────────┘   │ ┌──────────────────┐ │
│ │CameraPipe │ │    WarpedFrameRef           │ │ RGA 主体拷贝      │ │
│ │(V4L2+MPP) │─┼──► (时间戳匹配的帧组)       │ │ (95.7% blit)     │ │
│ ├───────────┤ │              │              │ ├──────────────────┤ │
│ │GpuWarpRoi │ │              ▼              │ │ GPU 接缝融合      │ │
│ │(OpenCL    │ │   ┌──────────────────┐      │ │ (3×34px Y/UV)    │ │
│ │ remap)    │ │   │ PanoramaComposer │      │ └──────────────────┘ │
│ └───────────┘ │   │  compose()       │      │ 三槽 NV12 全景池     │
└───────────────┘   │  convert_bgr()   │      │ + 六槽 BGR 租赁池    │
                    └────────┬─────────┘      └──────────┬───────────┘
                             │                           │
                             ▼                           ▼
                    PanoramaFrameRef            read() 兼容接口
                    (租赁式: acquire/release)   PanoramaFrame
```

**模块职责 / Module Responsibilities**

| 模块 | 职责 |
|---|---|
| `PanoramaPipeline` | 门面：生命周期管理（init/start/acquire/read/stop/stats） |
| `WarpProducer` ×4 | 每路一个线程：采集→解码→GPU warp 完整处理链 |
| `CameraPipe` | V4L2 DMA-BUF 采集 + MPP MJPEG→NV12 解码 |
| `GpuWarpRoi` | Mali OpenCL remap：按预标定 LUT 几何变换，`clImportMemoryARM` 导入 DMA-BUF |
| `FrameSynchronizer` | 按 V4L2 时间戳配对四路，超时丢旧帧不积压 |
| `PanoramaComposer` | RGA 主体拷贝（95.7%）+ GPU 窄接缝融合（4.3%）+ RGA 裁剪转 BGR |
| 三槽 NV12 全景池 | 全景中间输出槽状态机，防止读写冲突 |
| 六槽 BGR 租赁池 | 最终输出槽：`acquire()` 租用、`release()` 归还，支持异步下游消费 |

## 🎯 标定与资产生成 / Calibration & Asset Generation

拼接的正确性来源于**离线预标定**。相机参数和映射表在 PC 端用 Python 一次性生成并冻结，板端运行时只查表、零几何计算。

### 标定流程 / Pipeline

```text
① 四相机 fisheye 内参标定 (OpenCV fisheye K/D 模型)
        │  RMS: cam1 1.389 / cam2 0.741 / cam3 1.464 / cam4 1.114
        │  （标定脚本与内参在 calibration/ 目录，随仓库提供）
        ▼
② open-chain 投影设计 (cam2→cam1→cam4→cam3, yaw-only 90° 排列)
        │  FOV 150°, 2048 px/360° 密度, 输出 ~2389×720
        │  （stitch_360.py: 拼接算法与投影实现）
        ▼
③ 自动接缝搜索 + 羽化 (34px feather, 接缝带宽 34–36px)
        ▼
④ export_open_chain_assets.py 离线导出
        │  map_x/map_y (f32) + valid (u8) + seam 权重 + manifest
        │  （导出脚本在 calibration/ 目录）
        ▼
⑤ 冻结为 assets/open_chain_v1/ (SHA-256 校验)
        ▼
⑥ 板端 GPU remap 运行时只查表 (初始化时上传一次)
```

### 关键设计 / Key Design

- **fisheye 内参**：四相机均 1280×720，OpenCV fisheye K/D 模型，标定 RMS 0.741–1.464
- **open-chain 投影**：420° 非闭合视场（`cam2→cam1→cam4→cam3`），避免首尾闭合的接缝失真
- **确定性输出**：映射表是固定的（非逐帧计算），GPU remap 输出可复现——1000 次静态测试 hash 完全一致
- **自动接缝**：`seams_x = [754, 1156, 1711]`，feather 34px、接缝带宽 34–36px，Y/UV 分别存左右权重
- **manifest.json**：记录每个文件的 SHA-256、valid 像素比、ROI 偏移，运行前可校验资产完整性
- **相机身份**：按 USB Hub 下游端口解析（`1-1.4.<port>`），不依赖易变的 `/dev/videoN`
- **标定工具随仓库提供**：`calibration/` 目录包含标定脚本（`stitch_360.py`、`export_open_chain_assets.py`）和四路内参（`calib_cam1~4.npz`），可在 PC 端重新标定并重新导出资产

**标定脚本用法 / Calibration Script Usage**：

```bash
# 1. 确认 PC 端依赖（Python 3 + NumPy + OpenCV fisheye/V4L2）
python3 -c "import cv2, numpy; print(cv2.__version__)"

# 2. 接入四路摄像头，确认 /dev/videoN 对应 cam2→cam1→cam4→cam3
v4l2-ctl --list-devices

# 3. 实时标定预览（open-chain 投影 + 自动接缝）
python3 calibration/stitch_360.py --live --devices 0 1 2 3 --open-chain --auto-seams

# 4. 导出冻结资产（golden 参考目录 → 输出目录）
python3 calibration/export_open_chain_assets.py \
  --golden-dir <golden_images_dir> \
  --output-dir assets/open_chain_v1
```

> 导出命令的详细参数见 `python3 calibration/export_open_chain_assets.py --help`。

> ⚠️ 相机、镜头方向或分辨率变化后必须重新标定，不能直接复用现有参数。

## 📁 目录结构 / Repository Layout

```text
panorama_pipeline/
├── apps/
│   ├── panorama_display.cpp           # 视觉验收显示程序
│   └── panorama_output_lease_test.cpp # 输出槽租赁测试
├── include/                           # 公共头文件
│   ├── panorama_pipeline.h            # 对外 API（init/start/read）
│   ├── warp_producer.h                # 单路采集+处理 Worker
│   ├── frame_synchronizer.h           # 四路帧同步
│   ├── panorama_composer.h            # 全景合成器
│   ├── gpu_warp_roi.h                 # GPU ROI 几何变换
│   └── ...
├── src/                               # 实现（无 OpenCV 依赖）
│   ├── v4l2_streaming.cpp             # V4L2 DMA-BUF 采集
│   ├── mpp.cpp                        # MPP 硬解码
│   ├── gpu_warp_roi.cpp               # OpenCL remap kernel
│   ├── panorama_composer.cpp          # RGA 主体 + GPU 接缝
│   ├── frame_synchronizer.cpp         # 时间戳配对
│   └── ...
├── assets/
│   └── open_chain_v1/                 # 预标定映射表 + 接缝权重
├── calibration/                       # 标定工具（PC 端重新标定用）
│   ├── stitch_360.py                  # 拼接算法 + open-chain 投影
│   ├── export_open_chain_assets.py    # 离线导出映射表资产
│   ├── calib_cam1~4.npz               # 四路 fisheye 内参
│   └── camera_order.json              # 相机逻辑顺序
├── tests/
│   └── artifacts/                     # 测试输出
├── Makefile
└── README.md
```

## 📖 文件解析 / File Guide

> 本节按**数据流顺序**逐个文件解释作用，帮助理解管线如何从摄像头图像变成全景输出。

### 采集与解码（Input & Decode）

| 文件 | 作用 |
|---|---|
| `src/v4l2_streaming.cpp` | **V4L2 采集**。打开 `/dev/videoN` 摄像头，配置 MJPG 格式，用 `mmap`/DMA-BUF 方式申请采集缓冲，循环出队图像帧（只传 fd，不拷贝像素） |
| `src/camera_resolver.cpp` | **相机身份解析**。通过 USB Hub 下游端口号（`1-1.4.<port>`）识别每路相机的 `/dev/videoN`，不依赖易变的设备节点顺序 |
| `include/camera_resolver.h` | 相机解析函数声明（`resolve_usb_camera_by_hub_port`） |
| `src/mpp.cpp` | **MPP 硬解码**。调用 Rockchip MPP 把 MJPG 帧解码为 NV12（YUV420 半平面），解码输出直接进 DMA-BUF，CPU 不碰像素 |
| `include/mpp.h` | MPP 解码上下文与接口（`MppDecCtx`、`mpp_dec_*` 系列） |
| `src/camer_pip.cpp` | **单路相机管线**。把 V4L2 采集 + MPP 解码封装成一条流水线（`cp_open`/`cp_next`/`cp_close`），WarpProducer 直接调用 |
| `include/camer_pip.h` | 相机管线接口（`CameraPipe` 结构 + `cp_*` 函数） |

### GPU 几何变换（GPU Warp）

| 文件 | 作用 |
|---|---|
| `src/gpu_warp_roi.cpp` | **OpenCL remap 核心**（538 行）。初始化 Mali GPU OpenCL context，用 `clImportMemoryARM` 导入 DMA-BUF，加载预标定 map_x/map_y/valid 表，Y/UV 双 kernel 把每路相机图像按映射表变换到全景坐标系，输出 NV12 到 DMA-BUF |
| `include/gpu_warp_roi.h` | GPU warp 配置与接口（`GpuWarpRoiConfig`、`GpuWarpRoi`） |

### 单路 Worker 与帧同步（Producer & Sync）

| 文件 | 作用 |
|---|---|
| `src/warp_producer.cpp` | **单路 Warp Producer**（410 行）。每路相机一个：内部维护 3 槽缓冲池（`FREE→WRITING→READY→READING` 状态机），worker 线程循环采集→解码→GPU warp，`acquire_latest()` 供下游取最新帧 |
| `include/warp_producer.h` | Producer 配置/统计/帧引用接口（`WarpConfig`、`WarpProducerStats`、`WarpedFrameRef`） |
| `src/frame_synchronizer.cpp` | **四路帧同步**。从 4 个 Producer 各取一帧，按时间戳配对成组，超过 `maximum_spread_ns`（40ms）的旧帧被拒，保证四路画面时间对齐 |
| `include/frame_synchronizer.h` | 同步组结构与接口（`SynchronizedFrameGroup`、`FrameSynchronizer`） |

### 全景合成（Compose & Output）

| 文件 | 作用 |
|---|---|
| `src/panorama_composer.cpp` | **全景合成器**（399 行）。RGA 把 4 路 warp 输出按 `kInputs[4]` 布局拷到全景画布（主体 95.7%），OpenCL 3 条接缝（36/36/34px）融合，最后 RGA 裁剪 NV12→BGR888 输出 |
| `include/panorama_composer.h` | 合成器常量（`kBgrWidth=2248`、`kBgrHeight=330`、`kBgrStride=2256`、`kOutputSlots=3`、`kBgrSlots=6`）与接口 |
| `src/panorama_pipeline.cpp` | **对外 API 实现**（454 行）。`init(assets)`→`start()`→`acquire()/read()`，管理 4 个 Producer + 同步器 + 合成器 + BGR 6 槽租赁池，`release_callback` 支持异步下游 |
| `include/panorama_pipeline.h` | 对外 API（`PanoramaPipeline`、`PanoramaFrameRef`、`PanoramaFrame`、`PanoramaPipelineStats`） |

### 底层支撑（Foundation）

| 文件 | 作用 |
|---|---|
| `src/dma_alloc.cpp` | **DMA-BUF 分配**。封装 `/dev/dma_heap/system-uncached-dma32` 分配 + mmap 映射 + cache 同步，全管线共享同一套零拷贝内存 |
| `include/dma_alloc.h` | DMA 分配接口（`dma_alloc`/`dma_free`/`dma_sync_cpu_to_device` 等） |

### 应用程序（apps/）

| 文件 | 作用 |
|---|---|
| `apps/panorama_display.cpp` | **视觉验收显示程序**。启动管线，`acquire()` 取帧，用 OpenCV 显示全景画面（生产库不依赖 OpenCV，仅此程序用于验收） |
| `apps/panorama_output_lease_test.cpp` | **输出槽租赁测试**。验证 BGR 6 槽池的 acquire/release 契约：6 个唯一 fd、第 7 个阻塞、槽位复用、`read()` 兼容接口（产出 `passed=1` 证据） |

### 资产与标定（Assets & Calibration）

| 文件 | 作用 |
|---|---|
| `assets/open_chain_v1/` | **预标定资产**（32 文件，16MB）。map_x/map_y（f32 映射表）+ valid 掩码 + 3 条接缝的 Y/UV 左右权重 + coverage + manifest.json（SHA-256 校验） |
| `calibration/stitch_360.py` | **标定算法**。OpenCV fisheye 标定 + open-chain 投影 + 自动接缝搜索（PC 端运行） |
| `calibration/export_open_chain_assets.py` | **资产导出**。把标定结果导出为板端格式（map_x/map_y f32、valid u8、接缝权重、manifest） |
| `calibration/calib_cam1~4.npz` | 四路相机 fisheye 内参（K/D 矩阵） |
| `calibration/camera_order.json` | 相机逻辑顺序（cam2→cam1→cam4→cam3） |

### 构建配置

| 文件 | 作用 |
|---|---|
| `Makefile` | 构建脚本：`CORE_NAMES` 声明 10 个核心源文件 → 静态库 `libpanorama_pipeline.a`；`make all` 编译，`make check-libs` 校验 Mali OpenCL 链接路径，`make clean` 清理 |

## 📦 输出契约 / Output Contract

`PanoramaPipeline::acquire()` 返回 `PanoramaFrameRef`（租赁式），`read()` 返回 `PanoramaFrame`（兼容接口），均包含：

- **格式**：BGR888（DMA-BUF 后端）
- **宽高**：2248 × 330（`kBgrWidth × kBgrHeight`）
- **stride**：2256 像素（`kBgrStride`）
- **字节数**：`2256 × 330 × 3`
- **来源**：内部 NV12 全景（2390×720，stride 2432）裁剪 `(84, 198, 2248, 330)`
- **元数据**：`sequence`（帧序号）、`timestamp_ns`（组内最大时间戳）、`camera_spread_ns`（四路时间戳离散度）

**所有权语义 / Ownership**：

- `acquire()` 租用 BGR 输出槽（六槽池），调用方必须 `release()` 归还，或把 `release_callback()`/`release_context()` 转移给异步下游消费者（如 NPU 推理完成后归还）
- `read()` 为兼容接口，内部持有一个 lease，内容在下一次成功 `read()` 前有效

裁剪 + NV12→BGR 由**一次 1:1 RGA 操作**完成（`convert_bgr()`），生产链路无任何图像缩放。

## 🔧 构建与运行 / Build & Run

### 📋 复刻条件 / Reproduction Requirements

#### 硬件 / Hardware

| 组件 | 规格 | 备注 |
|---|---|---|
| 开发板 | **Rockchip RK3588 开发板** | 需 Mali GPU + NPU + MPP/VPU |
| 摄像头 ×4 | USB 摄像头，1280×720 MJPG@30 | 需接入同一 USB Hub（下游端口 2/3/4/5） |
| 存储 | eMMC 或 SD 卡 | 系统盘 |
| 电源 | 板卡配套电源 | 四路摄像头供电稳定 |

#### 软件环境 / Software Environment

| 组件 | 版本/要求 | 备注 |
|---|---|---|
| 操作系统 | Debian/Ubuntu (aarch64) | RK3588 板预装系统 |
| Linux 内核 | 5.10（Rockchip BSP） | 需 dma-heap 支持 |
| 编译器 | GCC (C++17) | `g++ -std=c++17` |
| CMake/Make | make ≥ 4.x | 本项目使用 Makefile |
| MPP | **自编译版**（官方源码） | 见下方编译步骤 |
| RGA | librga（系统自带或官方源码） | `librga.so` |
| Mali OpenCL | `/usr/lib/aarch64-linux-gnu/libmali-x11/` | 需 `clImportMemoryARM` |
| dma-heap | `/dev/dma_heap/cma-uncached` | 内核需启用 |

#### 标定资产 / Calibration Assets

`assets/open_chain_v1/` 已随仓库提供（16 MB）。包含：

- 四路相机的 `map_x/map_y`（f32 浮点映射表）+ `valid` 掩码
- 三条接缝的 Y/UV 左右权重 + coverage 掩码
- `manifest.json`（SHA-256 校验清单）

**重新标定**：`calibration/` 目录提供完整标定工具链（`stitch_360.py` 拼接算法、`export_open_chain_assets.py` 资产导出、四路内参 `calib_cam1~4.npz`），可在 PC 端完成重新标定与资产导出。

> ℹ️ **标定工具为本项目原创**：`calibration/` 下的 Python 脚本是本项目自研的离线标定与资产导出工具（OpenCV fisheye 标定 + 自研 open-chain 投影与接缝算法），非第三方代码。PC 端需安装 `opencv-python`、`numpy`。

> ⚠️ 如果相机型号/镜头/安装方向与标定时不同，**必须重新标定**并重新导出资产（见"标定与资产生成"章节）。

### 依赖 / Dependencies

| 组件 | 说明 |
|---|---|
| Rockchip RK3588 开发板 | 本工程在 RK3588 板验证 |
| V4L2 摄像头 ×4 | 1280x720 MJPG@30 |
| MPP (librockchip_mpp) | Rockchip 媒体处理库，**自编译版**（见下方说明） |
| RGA (librga) | Rockchip 2D 加速 |
| Mali OpenCL | `libmali-x11`，需支持 `clImportMemoryARM` |
| dma-heap | `/dev/dma_heap/cma-uncached` |

> ⚠️ **MPP 必须使用自编译版本**：系统自带的 `/usr/lib/aarch64-linux-gnu/librockchip_mpp.so`（2024-07 编译）存在 MJPEG 解码 segfault bug（`mpp_dec_decode` 内部崩溃）。本项目从 [Rockchip MPP 官方源码](https://github.com/rockchip-linux/mpp) 重新编译并安装到 `/usr/local`（不动系统库）：
>
> ```sh
> git clone https://github.com/rockchip-linux/mpp.git
> cd mpp/build/linux/aarch64/
> ./make-Makefiles.bash
> make -j$(nproc)
> sudo make install   # 安装到 /usr/local
> ```
>
> Makefile 中 `-I/usr/local/include` 和 `-L/usr/local/lib` 即指向自编译安装路径。

> ⚠️ **OpenCL 必须链接** `/usr/lib/aarch64-linux-gnu/libmali-x11/libOpenCL.so.1`（SONAME `libmali.so.1`），不能链接通用 ICD loader，否则缺少 `clImportMemoryARM`。

### 构建 / Build

```sh
make -j4
make check-libs        # 检查库链接是否正确
```

### 运行（视觉验收）/ Run (visual acceptance)

```sh
sudo env DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  ./build/panorama_display assets/open_chain_v1
```

按 `q` 或 `Esc` 退出。

### 🚀 完整复刻步骤 / Full Reproduction Steps

从零开始复刻整个项目的推荐顺序：

```text
① 准备硬件：RK3588 开发板 + 4×USB 摄像头（接入同一 Hub）
② 安装系统：Debian/Ubuntu aarch64 + Rockchip BSP 内核 5.10
③ 编译安装自编译 MPP（见上方"MPP 必须使用自编译版本"）
④ 确认依赖：librga、libmali-x11（含 clImportMemoryARM）、dma-heap
⑤ 克隆本仓库：git clone <repo-url>
⑥ 构建：make -j4 && make check-libs
⑦ 校验资产：确认 assets/open_chain_v1/manifest.json 存在且完整
⑧ 接入四路摄像头，按 USB Hub 端口确认相机身份
⑨ 运行：sudo ./build/panorama_display assets/open_chain_v1
⑩ 验收：观察全景画面，确认无错位、无黑边、接缝平滑
```

> 💡 **排错提示**：如果启动失败，先运行 `make check-libs` 检查链接；再确认 `/dev/dma_heap/cma-uncached` 存在（需 sudo 运行）；最后确认摄像头节点与 USB Hub 端口匹配。

### ❓ 常见问题 / FAQ

| 问题 | 原因 | 解决 |
|---|---|---|
| `clImportMemoryARM` 未定义 | 链接了通用 ICD loader | 改用 `/usr/lib/aarch64-linux-gnu/libmali-x11/libOpenCL.so.1` |
| MPP 解码 segfault | 系统自带 MPP 库有 bug | 从官方源码重编并安装到 `/usr/local` |
| 启动报 dma-heap 权限错误 | dma-heap 需要 root | `sudo` 运行 |
| 画面错位/黑边 | 相机与标定时不一致 | 重新标定并导出资产 |
| 找不到摄像头 | `/dev/videoN` 变化 | 按 USB Hub 端口解析（代码内置） |
| 帧率上不去 | 摄像头输入帧率限制 | 换更高帧率摄像头（60/120 FPS） |

### 集成 API / Integration API

```cpp
#include "panorama_pipeline.h"

PanoramaPipeline pipeline;
pipeline.init(assets_dir);    // 初始化（加载标定资产）
pipeline.start();             // 启动四路 Worker 线程

// 推荐：租赁式 API（acquire/release）
PanoramaFrameRef frame;
if (pipeline.acquire(&frame, /*timeout_ms=*/1000)) {
    // frame.dma_fd() / frame.data() 是 BGR888 DMA-BUF
    // 使用完毕后 release() 归还槽位；
    // 或把 release_callback()/release_context() 转移给异步下游消费者
    frame.release();
}

// 兼容：read() 接口（内部持有一个 lease）
PanoramaFrame legacy;
pipeline.read(&legacy, 1000);

// 运行时统计
PanoramaPipelineStats stats = pipeline.stats();
// stats.frames / stats.timeouts / stats.errors / stats.rga_body_average_ms ...

pipeline.stop();
```

## 📊 性能数据 / Performance

| 输入源帧率 | 实测全景帧率 | 说明 |
|---|---:|---|
| 30 FPS 摄像头 | **29.96–30.03 FPS** | 跑满输入源，受输入帧率限制 |
| 60 FPS 摄像头 | **~50 FPS** | 管线吞吐余量充足 |
| 120 FPS 摄像头 | **~110 FPS** | 高帧率输入下仍保持高吞吐 |

> 管线实际吞吐能力远超 30 FPS。帧率瓶颈在输入源帧率，不在处理管线。

### 资源占用（30 FPS 输入下）

| 指标 | 数值 | 条件 |
|---|---|---|
| CPU 占用 | ~24.7–24.8% | 八核 SoC 口径 |
| GPU 占用 | ~11% | 1000 MHz |
| DDR 占用 | ~11–13% | 2112 MHz |
| RGA 主体拷贝 | ~1.728 ms/帧 | — |
| GPU 接缝融合 | ~0.970 ms/帧 | — |
| 整幅预览转换 | ~1.075 ms/帧 | RGA NV12→BGR |
| 系统内存 | ~33% | 实时进程 RSS ~167 MiB |

> 数据采集于 RK3588 开发板，四路 USB 摄像头，预热后持续采样。

## 🧪 测试 / Tests

```sh
make check-libs
# 输出槽租赁测试
./build/panorama_output_lease_test
```

- 静态接缝融合测试：连续 1000 次输出无漂移（hash `b2ae3d8cc078cd25`），三条接缝六 kernel 平均 1.216 ms
- 1000 帧实时验证：错误计数为 0
- 600 帧完整链路：预热后 29.96–30.03 FPS，timeout/compose error 均为 0

## 🔗 相关项目 / Related Projects

本项目是全景拼接 + NPU 检测系统的**核心拼接模块**，配套项目：

- [zero-copy-tri-core-npu-inference](https://github.com/yyyy231209/zero-copy-tri-core-npu-inference) — 三核 NPU 隧道裂缝检测（消费本项目的 DMA-BUF 输出，PackedStrips 一次推理）
- [rockchip-linux/mpp](https://github.com/rockchip-linux/mpp) — Rockchip 媒体处理库（本项目 MPP 自编译版）
- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — RKNN 模型转换与部署工具链（NPU 模块用）

## 📄 License

[MIT](LICENSE)

## 🙏 致谢 / Acknowledgements

- [Rockchip MPP](https://github.com/rockchip-linux/mpp) — 媒体处理库
- [Rockchip RGA](https://github.com/rockchip-linux/librga) — 2D 加速库
- Mali OpenCL — GPU 计算
