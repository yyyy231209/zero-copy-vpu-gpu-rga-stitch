#ifndef CAMERA_PIPE_H
#define CAMERA_PIPE_H

#include <stddef.h>
#include <stdint.h>
#include <sys/time.h>

#include "v4l2_streaming.h"
#include "mpp.h"

typedef enum {
    CP_NEXT_OK = 0,
    CP_NEXT_TIMEOUT = 1,
    CP_NEXT_DROP = 2,
    CP_NEXT_FATAL = -1,
} CameraPipeResult;

typedef struct {
    const char  *dev;
    int          fd;
    MjpegFrame   buffers[V4L2_CAPTURE_BUFFER_COUNT];
    int          w;
    int          h;
    int          sizeimage;
    int          streaming;
    int          mpp_ready;
    int          fatal;
    uint32_t     last_sequence;
    struct timeval last_timestamp;
    int          have_frame_metadata;
    MppDecCtx    dec;
} CameraPipe;

int   cp_open(CameraPipe *cp, const char *dev);
int   cp_next_status(CameraPipe *cp);
void *cp_next(CameraPipe *cp);
void  cp_close(CameraPipe *cp);

#endif
