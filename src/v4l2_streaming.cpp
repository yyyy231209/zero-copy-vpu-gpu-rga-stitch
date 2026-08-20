#include "v4l2_streaming.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define V4L2_DEQUEUE_TIMEOUT_MS 200

static int configure_capture_timing(int fd, const char *dev)
{
    struct v4l2_control exposure_priority;
    memset(&exposure_priority, 0, sizeof(exposure_priority));
    exposure_priority.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    exposure_priority.value = 0;
    if (ioctl(fd, VIDIOC_S_CTRL, &exposure_priority) < 0) {
        fprintf(stderr,
                "%s: cannot disable exposure_auto_priority; "
                "30 FPS cannot be guaranteed\n",
                dev);
        perror("VIDIOC_S_CTRL(V4L2_CID_EXPOSURE_AUTO_PRIORITY)");
        return -1;
    }

    struct v4l2_streamparm requested;
    memset(&requested, 0, sizeof(requested));
    requested.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    requested.parm.capture.timeperframe.numerator = 1;
    requested.parm.capture.timeperframe.denominator = PIX_FPS;
    if (ioctl(fd, VIDIOC_S_PARM, &requested) < 0) {
        perror("VIDIOC_S_PARM");
        return -1;
    }

    struct v4l2_streamparm actual;
    memset(&actual, 0, sizeof(actual));
    actual.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_PARM, &actual) < 0) {
        perror("VIDIOC_G_PARM");
        return -1;
    }

    const unsigned int numerator =
        actual.parm.capture.timeperframe.numerator;
    const unsigned int denominator =
        actual.parm.capture.timeperframe.denominator;
    if (numerator == 0 || denominator == 0 ||
        (unsigned long long)denominator !=
            (unsigned long long)PIX_FPS * numerator) {
        fprintf(stderr, "%s: requested %d FPS, driver returned %u/%u FPS\n",
                dev, PIX_FPS, denominator, numerator);
        errno = EINVAL;
        return -1;
    }

    printf("%s: exposure_auto_priority=0, frame rate=%u/%u FPS\n",
           dev, denominator, numerator);
    return 0;
}

static void reset_frame(MjpegFrame *frame)
{
    memset(frame, 0, sizeof(*frame));
    frame->dma_fd = -1;
}

static void reset_buffers(MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT])
{
    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i)
        reset_frame(&buffers[i]);
}

static void free_buffers(MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT])
{
    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i) {
        if (buffers[i].dma_fd >= 0 && buffers[i].dma_va)
            dma_buf_free(buffers[i].capacity, &buffers[i].dma_fd, buffers[i].dma_va);
        reset_frame(&buffers[i]);
    }
}

static void release_driver_buffers(int fd)
{
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;
    req.count = 0;
    (void)ioctl(fd, VIDIOC_REQBUFS, &req);
}

int v4l2_capture_init(const char *dev, int *width, int *height, int *sizeimage)
{
    if (!dev || !width || !height || !sizeimage) {
        errno = EINVAL;
        return -1;
    }

    int fd = open(dev, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) {
        perror("open camera");
        return -1;
    }

    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = PIX_WIDTH;
    fmt.fmt.pix.height = PIX_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        close(fd);
        return -1;
    }

    if (configure_capture_timing(fd, dev) < 0) {
        close(fd);
        return -1;
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
        perror("VIDIOC_G_FMT");
        close(fd);
        return -1;
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG ||
        fmt.fmt.pix.width == 0 || fmt.fmt.pix.height == 0 ||
        fmt.fmt.pix.sizeimage == 0) {
        fprintf(stderr, "camera did not accept a valid MJPEG capture format\n");
        close(fd);
        errno = EINVAL;
        return -1;
    }

    *width = (int)fmt.fmt.pix.width;
    *height = (int)fmt.fmt.pix.height;
    *sizeimage = (int)fmt.fmt.pix.sizeimage;

    printf("width = %d\n", *width);
    printf("height = %d\n", *height);
    const unsigned char *p = (const unsigned char *)&fmt.fmt.pix.pixelformat;
    printf("pixelformat = %c%c%c%c\n", p[0], p[1], p[2], p[3]);
    return fd;
}

int v4l2_capture_requeue(int fd,
                         const MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                         int index)
{
    if (fd < 0 || !buffers || index < 0 || index >= V4L2_CAPTURE_BUFFER_COUNT ||
        buffers[index].dma_fd < 0) {
        errno = EINVAL;
        return -1;
    }

    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_DMABUF;
    buffer.index = (unsigned int)index;
    buffer.m.fd = buffers[index].dma_fd;

    if (ioctl(fd, VIDIOC_QBUF, &buffer) < 0) {
        perror("VIDIOC_QBUF");
        return -1;
    }
    return 0;
}

int v4l2_capture_start(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                       int sizeimage, int width, int height)
{
    if (fd < 0 || !buffers || sizeimage <= 0 || width <= 0 || height <= 0) {
        errno = EINVAL;
        return -1;
    }

    reset_buffers(buffers);
    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i) {
        if (dma_buf_alloc(CMA_HEAP_UNCACHE_PATH, (size_t)sizeimage,
                          &buffers[i].dma_fd, &buffers[i].dma_va) < 0) {
            fprintf(stderr, "V4L2: DMA-BUF allocation %d failed\n", i);
            free_buffers(buffers);
            return -1;
        }
        buffers[i].capacity = (size_t)sizeimage;
        buffers[i].width = width;
        buffers[i].height = height;
    }

    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;
    req.count = V4L2_CAPTURE_BUFFER_COUNT;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0 ||
        req.count < V4L2_CAPTURE_BUFFER_COUNT) {
        perror("VIDIOC_REQBUFS");
        release_driver_buffers(fd);
        free_buffers(buffers);
        return -1;
    }

    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i) {
        if (v4l2_capture_requeue(fd, buffers, i) < 0) {
            release_driver_buffers(fd);
            free_buffers(buffers);
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        release_driver_buffers(fd);
        free_buffers(buffers);
        return -1;
    }
    return 0;
}

int v4l2_capture_dequeue(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                         int *out_index)
{
    if (fd < 0 || !buffers || !out_index) {
        errno = EINVAL;
        return V4L2_CAPTURE_ERROR;
    }

    struct pollfd pfd;
    memset(&pfd, 0, sizeof(pfd));
    pfd.fd = fd;
    pfd.events = POLLIN | POLLPRI;

    int ret;
    do {
        ret = poll(&pfd, 1, V4L2_DEQUEUE_TIMEOUT_MS);
    } while (ret < 0 && errno == EINTR);

    if (ret == 0)
        return V4L2_CAPTURE_TIMEOUT;
    if (ret < 0) {
        perror("camera poll");
        return V4L2_CAPTURE_ERROR;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        fprintf(stderr, "camera poll error: revents=0x%x\n", pfd.revents);
        return V4L2_CAPTURE_ERROR;
    }

    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(fd, VIDIOC_DQBUF, &buffer) < 0) {
        if (errno == EAGAIN)
            return V4L2_CAPTURE_TIMEOUT;
        perror("VIDIOC_DQBUF");
        return V4L2_CAPTURE_ERROR;
    }

    if (buffer.index >= V4L2_CAPTURE_BUFFER_COUNT ||
        buffer.bytesused == 0 ||
        buffer.bytesused > buffers[buffer.index].capacity) {
        fprintf(stderr, "V4L2 invalid frame: index=%u bytesused=%u\n",
                buffer.index, buffer.bytesused);
        return V4L2_CAPTURE_ERROR;
    }

    MjpegFrame *frame = &buffers[buffer.index];
    frame->size = buffer.bytesused;
    frame->sequence = buffer.sequence;
    frame->timestamp = buffer.timestamp;
    *out_index = (int)buffer.index;
    return V4L2_CAPTURE_OK;
}

void v4l2_capture_stop(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT])
{
    if (fd >= 0) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0 && errno != EINVAL)
            perror("VIDIOC_STREAMOFF");
        release_driver_buffers(fd);
    }
    if (buffers)
        free_buffers(buffers);
}
