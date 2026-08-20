#ifndef MPP_DECODE_H
#define MPP_DECODE_H

#include <stddef.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_meta.h>
#include <rockchip/mpp_buffer.h>

#include "dma_alloc.h"

#define MPP_OUT_BUF_COUNT  8
#define MPP_ALIGN(x, a)    (((x) + ((a) - 1)) & ~((a) - 1))

typedef struct {
    MppApi    *mpi;
    MppCtx     ctx;
    MppFrame   frame;
    MppBuffer  out_buf;
} MppDecCtx;

int   mpp_dec_init(MppDecCtx *dec);
/*
 * Return values:
 *   0  input DMA-BUF is no longer referenced and output frame is ready.
 *  -1  decoder was reset; input DMA-BUF is safe to requeue, but frame is dropped.
 *  -2  decoder reset failed; caller must close before reusing the input DMA-BUF.
 */
int   mpp_dec_frame(MppDecCtx *dec, int dma_fd, size_t mjpeg_size, MppFrame *outframe);
void  mpp_dec_close(MppDecCtx *dec);
void *mpp_dec_get_nv12(MppDecCtx *dec);

#endif
