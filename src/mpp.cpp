#include "mpp.h"

#include <stdio.h>
#include <string.h>

static int reset_decoder(MppDecCtx *dec)
{
    if (!dec || !dec->mpi || !dec->ctx)
        return -1;

    MPP_RET ret = dec->mpi->reset(dec->ctx);
    if (ret != MPP_OK) {
        fprintf(stderr, "MPP reset failed: %d\n", ret);
        return -1;
    }
    return 0;
}

int mpp_dec_init(MppDecCtx *dec)
{
    if (!dec)
        return -1;

    memset(dec, 0, sizeof(*dec));

    RK_S64 input_timeout = 0;
    RK_S64 output_timeout = 0;
    RK_U32 out_fmt = 0;
    if (mpp_create(&dec->ctx, &dec->mpi) != MPP_OK)
        return -1;

    if (mpp_init(dec->ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG) != MPP_OK) {
        mpp_destroy(dec->ctx);
        dec->ctx = NULL;
        dec->mpi = NULL;
        return -1;
    }

    input_timeout = 1000;
    if (dec->mpi->control(dec->ctx, MPP_SET_INPUT_TIMEOUT, &input_timeout) != MPP_OK) {
        mpp_dec_close(dec);
        return -1;
    }

    /* One CameraPipe is deliberately synchronous: wait for this packet's frame. */
    output_timeout = 1000;
    if (dec->mpi->control(dec->ctx, MPP_SET_OUTPUT_TIMEOUT, &output_timeout) != MPP_OK) {
        mpp_dec_close(dec);
        return -1;
    }

    out_fmt = MPP_FMT_YUV420SP;
    if (dec->mpi->control(dec->ctx, MPP_DEC_SET_OUTPUT_FORMAT, &out_fmt) != MPP_OK) {
        mpp_dec_close(dec);
        return -1;
    }

    const int width = 1280;
    const int height = 720;
    RK_U32 hor_stride = MPP_ALIGN(width, 16);
    RK_U32 ver_stride = MPP_ALIGN(height, 16);
    size_t nv12_size = (size_t)hor_stride * ver_stride * 2;

    if (mpp_buffer_get(NULL, &dec->out_buf, nv12_size) != MPP_OK)
        goto fail;

    if (mpp_frame_init(&dec->frame) != MPP_OK)
        goto fail;

    mpp_frame_set_buffer(dec->frame, dec->out_buf);
    mpp_frame_set_width(dec->frame, width);
    mpp_frame_set_height(dec->frame, height);
    mpp_frame_set_hor_stride(dec->frame, hor_stride);
    mpp_frame_set_ver_stride(dec->frame, ver_stride);
    mpp_frame_set_fmt(dec->frame, MPP_FMT_YUV420SP);
    return 0;

fail:
    mpp_dec_close(dec);
    return -1;
}

void mpp_dec_close(MppDecCtx *dec)
{
    if (!dec)
        return;

    if (dec->frame) {
        mpp_frame_deinit(&dec->frame);
        dec->frame = NULL;
    }
    if (dec->out_buf) {
        mpp_buffer_put(dec->out_buf);
        dec->out_buf = NULL;
    }
    if (dec->ctx) {
        mpp_destroy(dec->ctx);
        dec->ctx = NULL;
    }
    dec->mpi = NULL;
}

void *mpp_dec_get_nv12(MppDecCtx *dec)
{
    return dec && dec->out_buf ? mpp_buffer_get_ptr(dec->out_buf) : NULL;
}

int mpp_dec_frame(MppDecCtx *dec, int dma_fd, size_t mjpeg_size, MppFrame *outframe)
{
    if (!dec || !dec->ctx || !dec->mpi || !dec->frame || !dec->out_buf ||
        !outframe || dma_fd < 0 || mjpeg_size == 0)
        return -1;

    *outframe = NULL;
    MppBuffer in_buf = NULL;
    MppPacket packet = NULL;
    MppPacket packet_ret = NULL;
    MppFrame frame_ret = NULL;
    MppMeta packet_meta = NULL;
    MppMeta frame_meta = NULL;
    MPP_RET ret = MPP_OK;
    int submitted = 0;
    int result = -1;

    MppBufferInfo in_info;
    memset(&in_info, 0, sizeof(in_info));
    in_info.type = MPP_BUFFER_TYPE_EXT_DMA;
    in_info.fd = dma_fd;
    in_info.size = mjpeg_size;

    ret = mpp_buffer_import(&in_buf, &in_info);
    if (ret != MPP_OK)
        goto done;

    ret = mpp_packet_init_with_buffer(&packet, in_buf);
    if (ret != MPP_OK)
        goto done;
    mpp_packet_set_length(packet, mjpeg_size);

    packet_meta = mpp_packet_get_meta(packet);
    if (!packet_meta ||
        mpp_meta_set_frame(packet_meta, KEY_OUTPUT_FRAME, dec->frame) != MPP_OK)
        goto done;

    ret = dec->mpi->decode_put_packet(dec->ctx, packet);
    if (ret != MPP_OK)
        goto done;
    submitted = 1;

    ret = dec->mpi->decode_get_frame(dec->ctx, &frame_ret);
    if (ret != MPP_OK || !frame_ret)
        goto done;

    if (frame_ret != dec->frame ||
        mpp_frame_get_errinfo(frame_ret) ||
        mpp_frame_get_discard(frame_ret) ||
        mpp_frame_get_info_change(frame_ret)) {
        fprintf(stderr, "MPP invalid output frame: frame=%p expected=%p err=%u discard=%u info_change=%u\n",
                frame_ret, dec->frame, mpp_frame_get_errinfo(frame_ret),
                mpp_frame_get_discard(frame_ret), mpp_frame_get_info_change(frame_ret));
        goto done;
    }

    frame_meta = mpp_frame_get_meta(dec->frame);
    if (!frame_meta ||
        mpp_meta_get_packet(frame_meta, KEY_INPUT_PACKET, &packet_ret) != MPP_OK ||
        packet_ret != packet) {
        fprintf(stderr, "MPP did not return the submitted input packet\n");
        goto done;
    }

    *outframe = dec->frame;
    result = 0;

done:
    if (result != 0 && submitted) {
        if (reset_decoder(dec) != 0)
            result = -2;
    }

    /*
     * dec->frame is the preallocated output frame owned by MppDecCtx.
     * It must not be deinitialized per frame. Any unexpected frame is owned
     * by MPP and is released only on the error path.
     */
    if (frame_ret && frame_ret != dec->frame)
        mpp_frame_deinit(&frame_ret);
    if (packet)
        mpp_packet_deinit(&packet);
    if (in_buf)
        mpp_buffer_put(in_buf);

    return result;
}
