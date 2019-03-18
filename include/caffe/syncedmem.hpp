#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));//如果在GPU模式下，用cudaMallocHost()来分配内存
    *use_cuda = true;
    return;
  }
#endif
  *ptr = malloc(size);//如果在CPU模式下，用malloc()来分配内存
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));//如果在GPU模式下，用cudaFreeHost()来释放内存
    return;
  }
#endif
  free(ptr);//如果在CPU模式下，用free()来释放内存
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  ~SyncedMemory();
  const void* cpu_data();//只读方式获取CPU上面的数据
  void set_cpu_data(void* data);//设置CPU上面的数据
  const void* gpu_data();//只读方式获取GPU上面的数据
  void set_gpu_data(void* data);//设置GPU上面的数据
  void* mutable_cpu_data();//获取CPU上面的数据，并且支持改变CPU上面的数据
  void* mutable_gpu_data();//获取GPU上面的数据，并且支持改变GPU上面的数据
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };//一些指向数据的指针位置的枚举值
  SyncedHead head() { return head_; }//返回值指向最新数据的指针位置的枚举值
  size_t size() { return size_; }//返回数据的大小

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void to_cpu();//表示数据复制到CPU
  void to_gpu();//表示数据复制到GPU
  void* cpu_ptr_;//指向CPU的指针，通过此指针访问CPU的数据
  void* gpu_ptr_;//指向GPU的指针，通过此指针访问GPU的数据
  size_t size_;//表示数据大小
  SyncedHead head_;//指向目前最新数据块的位置（没有初始化，在CPU上，在GPU上，CPU和GPU共享数据）
  bool own_cpu_data_;//是否有GPU数据
  bool cpu_malloc_use_cuda_;//检测CPU数据是否占用了GPU空间
  bool own_gpu_data_;//是否有GPU数据
  int gpu_device_;//查询GPU设备编号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
