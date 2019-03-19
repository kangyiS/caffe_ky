#include <vector>
#include "caffe/layers/conv_layer.hpp"

namespace caffe {
//计算卷积层的输出形状
template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();//卷积核大小  
  const int* stride_data = this->stride_.cpu_data();//步长  
  const int* pad_data = this->pad_.cpu_data();//pad 
  const int* dilation_data = this->dilation_.cpu_data();//卷积核膨胀  
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);//在这里获取输入blob的height与width？？？  
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;//在这里进行卷积核的扩展操作  
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;//在这里计算卷积过后生成的blob的高和宽
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();//读入卷积层的参数（权重），blobs_[0]存储的权重，而blobs_[1]存储的偏置  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();//读入bottom blob的data  
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {//这里的num_指的是batch_size，也就是说，一张一张图片的来
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {//如果启用了偏置
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);//那么加上偏置  
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();//读入权重参数  
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();//读入权重的梯度 
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();//获取每个top blob的梯度  
    const Dtype* bottom_data = bottom[i]->cpu_data();//获取每个bottom blob的数据  
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();//获取每个bottom blob的梯度  
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {//如果这个blob需要反传并且启用了偏置的话 
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();//获取该层偏置的梯度
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);//对于每张输入的原图片偏置梯度的反传 
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {//如果该blob需要反传权值梯度，则反传  
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {//如果该blob需要反传数据梯度，则反传  
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
