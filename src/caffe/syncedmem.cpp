#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

//to_cpu()函数描述了将最新数据复制到cpu上面的处理
inline void SyncedMemory::to_cpu() {
  switch (head_) {//首先查看一下数据的位置 
  case UNINITIALIZED://如果数据没有初始化，则将数据初始化  
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;//将最新数据的位置放置在cpu  
    own_cpu_data_ = true;//置cpu有数据
    break;
  case HEAD_AT_GPU://如果数据在gpu上面  
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);//首先在cpu上面创建空间
      own_cpu_data_ = true;//置cpu有数据  
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);//然后将数据从gpu复制到cpu
    head_ = SYNCED;//最后置最新数据的标志为共享  
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}
//to_gpu()函数描述了将最新数据复制到gpu上面的处理
inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {//首先查看一下数据的位置
  case UNINITIALIZED://如果数据没有初始化，则将数据初始化  
    CUDA_CHECK(cudaGetDevice(&gpu_device_));//获取gpu设备信息  
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));//在gpu上面开辟空间  
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;//将最新数据的位置放置在gpu  
    own_gpu_data_ = true;//置gpu有数据  
    break;
  case HEAD_AT_CPU://如果数据在cpu上面  
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));//获取gpu设备信息  
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));//在gpu上面开辟空间  
      own_gpu_data_ = true;//置gpu有数据  
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);//然后将数据从cpu复制到gpu  
    head_ = SYNCED;//置数据为共享的  
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {//访问cpu上面的的数据，返回const void*类型的指针  
  to_cpu();//在访问的时候调用上文的to_cpu()函数，需要同步最新数据  
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {//利用data指针设置cpu上面的数据  
  CHECK(data);
  if (own_cpu_data_) {//如果目前cpu上面有数据，则先释放  
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;//将指向cpu上数据的指针指向data  
  head_ = HEAD_AT_CPU;//将最新数据的标志置为在cpu上  
  own_cpu_data_ = false;//为什么这里要false呢？？？
}

const void* SyncedMemory::gpu_data() {//返回gpu上面的数据，返回const void*类型的指针  
#ifndef CPU_ONLY
  to_gpu();//在访问的时候调用上文的to_gpu()函数，需要同步最新数据 
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {//利用data指针设置gpu上面的数据  
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {//如果目前gpu上面有数据，则先释放  
    int initial_device;
    cudaGetDevice(&initial_device);//查询gpu设备信息  
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));//设置当前所使用的设备  
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));//释放gpu上面的数据  
    cudaSetDevice(initial_device);//再次设置当前所使用的设备  
  }
  gpu_ptr_ = data;//将指向gpu上数据的指针指向data  
  head_ = HEAD_AT_GPU;//将最新数据的标志置为在gpu上  
  own_gpu_data_ = false;//为什么这里要false呢？？？
#else
  NO_GPU;
#endif
}
//和前面的cpu_data()相比，调用这个函数之后，在cpu上面的数据可能会发生改变
void* SyncedMemory::mutable_cpu_data() { 
  to_cpu();//在访问的时候调用上文的to_cpu()函数，需要同步最新数据  
  /*在访问完毕cpu数据之后，将最新数据的指针置在了cpu，意味着若 
  要在gpu上面改变数据的时候，需要先同步一下cpu的数据*/  
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}
//和前面的gpu_data()相比，调用这个函数之后，在gpu上面的数据可能会发生改变
void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();//在访问的时候调用上文的to_gpu()函数，需要同步最新数据  
  /*在访问完毕gpu数据之后，将最新数据的指针置在了gpu，意味着若 
  要在cpu上面改变数据的时候，需要先同步一下gpu的数据*/  
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}
//以流的方式将数据同步到gpu
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);//检验最新数据位于cpu上面  
  if (gpu_ptr_ == NULL) {//如果当前gpu上面没有数据，则在gpu上面分配数据的存储空间  
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));//将cpu上面的数据同步至gpu  
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;//将最新数据的指针置为共享  
}
#endif

}  // namespace caffe

