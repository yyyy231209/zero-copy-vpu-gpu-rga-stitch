#include "dma_alloc.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned long long __u64;
typedef unsigned int __u32;

struct dma_heap_allocation_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};

#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)

struct dma_buf_sync {
    __u64 flags;
};

#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)

int dma_sync_device_to_cpu(int fd)
{
    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

int dma_sync_cpu_to_device(int fd)
{
    struct dma_buf_sync sync = {0};
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

int dma_buf_alloc(const char *path, size_t size, int *fd, void **va)
{
    if (!path || !fd || !va || size == 0) {
        errno = EINVAL;
        return -EINVAL;
    }

    *fd = -1;
    *va = NULL;

    int heap_fd = open(path, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        perror("open dma_heap");
        return -errno;
    }

    struct dma_heap_allocation_data allocation;
    memset(&allocation, 0, sizeof(allocation));
    allocation.len = size;
    allocation.fd_flags = O_CLOEXEC | O_RDWR;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &allocation) < 0) {
        int saved_errno = errno;
        perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        return -saved_errno;
    }

    void *mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, allocation.fd, 0);
    if (mapped == MAP_FAILED) {
        int saved_errno = errno;
        perror("mmap dma-buf");
        close(allocation.fd);
        close(heap_fd);
        return -saved_errno;
    }

    close(heap_fd);
    *fd = (int)allocation.fd;
    *va = mapped;
    return 0;
}

void dma_buf_free(size_t size, int *fd, void *va)
{
    if (va && va != MAP_FAILED && size > 0)
        (void)munmap(va, size);

    if (fd && *fd >= 0) {
        (void)close(*fd);
        *fd = -1;
    }
}
