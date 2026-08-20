#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "camer_pip.h"

static void camera_pipe_reset(CameraPipe *cp)
{
    memset(cp, 0, sizeof(*cp));
    cp->fd = -1;
    for (int i = 0; i < V4L2_CAPTURE_BUFFER_COUNT; ++i)
        cp->buffers[i].dma_fd = -1;
}

int cp_open(CameraPipe *cp, const char *dev)
{
    if (!cp || !dev) {
        fprintf(stderr, "CameraPipe: invalid open arguments\n");
        return -1;
    }

    camera_pipe_reset(cp);
    cp->dev = dev;

    cp->fd = v4l2_capture_init(dev, &cp->w, &cp->h, &cp->sizeimage);
    if (cp->fd < 0)
        goto fail;

    if (v4l2_capture_start(cp->fd, cp->buffers, cp->sizeimage, cp->w, cp->h) < 0)
        goto fail;
    cp->streaming = 1;

    if (mpp_dec_init(&cp->dec) < 0)
        goto fail;
    cp->mpp_ready = 1;

    return 0;

fail:
    cp_close(cp);
    return -1;
}

int cp_next_status(CameraPipe *cp)
{
    if (!cp || cp->fatal || !cp->streaming || !cp->mpp_ready)
        return CP_NEXT_FATAL;

    int index = -1;
    int capture_ret = v4l2_capture_dequeue(cp->fd, cp->buffers, &index);
    if (capture_ret == V4L2_CAPTURE_TIMEOUT)
        return CP_NEXT_TIMEOUT;
    if (capture_ret != V4L2_CAPTURE_OK) {
        cp->fatal = 1;
        return CP_NEXT_FATAL;
    }

    const uint32_t sequence = cp->buffers[index].sequence;
    const struct timeval timestamp = cp->buffers[index].timestamp;
    MppFrame out_frame = NULL;
    int decode_ret = mpp_dec_frame(&cp->dec, cp->buffers[index].dma_fd,
                                   cp->buffers[index].size, &out_frame);

    if (decode_ret == -2) {
        /* MPP could not reset; caller must close before reusing this DMA-BUF. */
        cp->fatal = 1;
        return CP_NEXT_FATAL;
    }

    if (v4l2_capture_requeue(cp->fd, cp->buffers, index) < 0) {
        cp->fatal = 1;
        return CP_NEXT_FATAL;
    }

    if (decode_ret < 0)
        return CP_NEXT_DROP;

    if (out_frame != cp->dec.frame || !mpp_dec_get_nv12(&cp->dec)) {
        fprintf(stderr, "CameraPipe: decoder returned an invalid output frame\n");
        cp->fatal = 1;
        return CP_NEXT_FATAL;
    }
    cp->last_sequence = sequence;
    cp->last_timestamp = timestamp;
    cp->have_frame_metadata = 1;
    return CP_NEXT_OK;
}

void *cp_next(CameraPipe *cp)
{
    return cp_next_status(cp) == CP_NEXT_OK ? mpp_dec_get_nv12(&cp->dec) : NULL;
}

void cp_close(CameraPipe *cp)
{
    if (!cp)
        return;

    if (cp->mpp_ready) {
        mpp_dec_close(&cp->dec);
        cp->mpp_ready = 0;
    }
    if (cp->streaming) {
        v4l2_capture_stop(cp->fd, cp->buffers);
        cp->streaming = 0;
    }
    if (cp->fd >= 0) {
        close(cp->fd);
        cp->fd = -1;
    }
    camera_pipe_reset(cp);
}
