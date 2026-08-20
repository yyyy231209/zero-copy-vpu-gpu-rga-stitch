#ifndef V4L2_STREAMING_H
#define V4L2_STREAMING_H

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>
#include <linux/videodev2.h>

#include "dma_alloc.h"

#define PIX_WIDTH 1280
#define PIX_HEIGHT 720
#define PIX_STRIDE 1280
#define PIX_FPS 30
#define V4L2_CAPTURE_BUFFER_COUNT 4

typedef struct {
    int            dma_fd;
    void          *dma_va;
    size_t         size;
    size_t         capacity;
    int            width;
    int            height;
    uint32_t       sequence;
    struct timeval timestamp;
} MjpegFrame;

typedef enum {
    V4L2_CAPTURE_OK = 0,
    V4L2_CAPTURE_TIMEOUT = 1,
    V4L2_CAPTURE_ERROR = -1,
} V4l2CaptureResult;

int  v4l2_capture_init(const char *dev, int *width, int *height, int *sizeimage);
int  v4l2_capture_start(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                         int sizeimage, int width, int height);
int  v4l2_capture_dequeue(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                           int *out_index);
int  v4l2_capture_requeue(int fd,
                           const MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT],
                           int index);
void v4l2_capture_stop(int fd, MjpegFrame buffers[V4L2_CAPTURE_BUFFER_COUNT]);

#endif
