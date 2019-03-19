#include <algorithm>  
#include <vector>  
  
#include "caffe/filler.hpp"  
#include "caffe/layers/base_conv_layer.hpp"  
#include "caffe/util/im2col.hpp"  
#include "caffe/util/math_functions.hpp"  
  
namespace caffe {  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  // Configure the kernel size, padding, stride, and inputs.  
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();//读入参数  
  force_nd_im2col_ = conv_param.force_nd_im2col();//读入标志进行强制n维卷积的参数  
  /*channel_axis_这个参数读取参数定义中的axis参数，默认为1，表示按channel求和，输入blob为(N,C,W,H)时， 
  一个输出通道对应的所有卷积核对输入blob上各通道做二维卷积，最后将输入各通道卷积的结果加起来，作为 
  一张输出的特征子图*/  
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());  
  const int first_spatial_axis = channel_axis_ + 1;//指示卷积输入图像的第一个轴，往往是H(height)  
  const int num_axes = bottom[0]->num_axes();//得到bottom blob的维度  
  num_spatial_axes_ = num_axes - first_spatial_axis;//卷积处理的维度数  
  CHECK_GE(num_spatial_axes_, 0);//卷积处理的维度数必须大于0  
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);//用于初始化卷积操作输入数据的形状，一般三维(C,H,W)  
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));//用于初始化卷积核的形状  
  // Setup filter kernel dimensions (kernel_shape_).  
  kernel_shape_.Reshape(spatial_dim_blob_shape);//初始化卷积核的形状(高*宽)  
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();//得到记录卷积核形状数据地址  
  /*检查参数中有没有自定义二维卷积的卷积核长宽，如果有定义则分别赋值，且自定义了二维卷积核 
  长宽的话，kernal_size参数将不能被定义，否则非法。若参数中没有定义二维卷积核的长宽，那么根据 
  kernal_size参数给卷积核赋值，卷积核一般是正方形*/  
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {  
    CHECK_EQ(num_spatial_axes_, 2)  
        << "kernel_h & kernel_w can only be used for 2D convolution.";  
    CHECK_EQ(0, conv_param.kernel_size_size())  
        << "Either kernel_size or kernel_h/w should be specified; not both.";  
    kernel_shape_data[0] = conv_param.kernel_h();  
    kernel_shape_data[1] = conv_param.kernel_w();  
  } else {  
    const int num_kernel_dims = conv_param.kernel_size_size();  
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)  
        << "kernel_size must be specified once, or once per spatial dimension "  
        << "(kernel_size specified " << num_kernel_dims << " times; "  
        << num_spatial_axes_ << " spatial dims).";  
      for (int i = 0; i < num_spatial_axes_; ++i) {  
        kernel_shape_data[i] =  
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);  
      }  
  }  
  //检查卷积核参数(高宽)是否合法  
  for (int i = 0; i < num_spatial_axes_; ++i) {  
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";  
  }  
  // Setup stride dimensions (stride_).  
  stride_.Reshape(spatial_dim_blob_shape);//初始化步长，注意，卷积核处理二维图像的话，步长也是二维的  
  int* stride_data = stride_.mutable_cpu_data();//得到卷积核步长参数的地址  
  /*检查参数中有没有自定义二维卷积时高和宽方向的步长，如果定义了则赋值。如果没有定义的话，就按照我们 
  定义的网络参数文件中的卷积层的stride参数赋值，stride参数要是缺失的话步长默认为kDefaultStride，即为1， 
  我们往往只定义了一个步长值，代表高和宽方向的步长一致。*/  
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {  
    CHECK_EQ(num_spatial_axes_, 2)  
        << "stride_h & stride_w can only be used for 2D convolution.";  
    CHECK_EQ(0, conv_param.stride_size())  
        << "Either stride or stride_h/w should be specified; not both.";  
    stride_data[0] = conv_param.stride_h();  
    stride_data[1] = conv_param.stride_w();  
  } else {  
    const int num_stride_dims = conv_param.stride_size();  
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||  
          num_stride_dims == num_spatial_axes_)  
        << "stride must be specified once, or once per spatial dimension "  
        << "(stride specified " << num_stride_dims << " times; "  
        << num_spatial_axes_ << " spatial dims).";  
    const int kDefaultStride = 1;  
    for (int i = 0; i < num_spatial_axes_; ++i) {  
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :  
          conv_param.stride((num_stride_dims == 1) ? 0 : i);  
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";  
    }  
  }  
  // Setup pad dimensions (pad_).  
  /*检查参数中有没有自定义高和宽方向的pad，如果定义了则赋值。如果没有定义的话，就按照我们 
  定义的网络参数文件中的卷积层的pad参数赋值，pad参数要是缺失的话默认为kDefaultPad，即为0， 
  我们往往只定义了一个pad值，代表高和宽方向的pad一致。*/  
  pad_.Reshape(spatial_dim_blob_shape);  
  int* pad_data = pad_.mutable_cpu_data();  
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {  
    CHECK_EQ(num_spatial_axes_, 2)  
        << "pad_h & pad_w can only be used for 2D convolution.";  
    CHECK_EQ(0, conv_param.pad_size())  
        << "Either pad or pad_h/w should be specified; not both.";  
    pad_data[0] = conv_param.pad_h();  
    pad_data[1] = conv_param.pad_w();  
  } else {  
    const int num_pad_dims = conv_param.pad_size();  
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||  
          num_pad_dims == num_spatial_axes_)  
        << "pad must be specified once, or once per spatial dimension "  
        << "(pad specified " << num_pad_dims << " times; "  
        << num_spatial_axes_ << " spatial dims).";  
    const int kDefaultPad = 0;  
    for (int i = 0; i < num_spatial_axes_; ++i) {  
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :  
          conv_param.pad((num_pad_dims == 1) ? 0 : i);  
    }  
  }  
  /*检查参数中有没有自定义高和宽方向的卷积核扩展，如果定义了则赋值。如果没有定义的话，就按照我们 
  定义的网络参数文件中的卷积层的dilation参数赋值，dilation_参数要是缺失的话默认为kDefaultDilation， 
  即为1，表示卷积核不进行扩展。*/  
  // Setup dilation dimensions (dilation_).  
  dilation_.Reshape(spatial_dim_blob_shape);  
  int* dilation_data = dilation_.mutable_cpu_data();  
  const int num_dilation_dims = conv_param.dilation_size();  
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||  
        num_dilation_dims == num_spatial_axes_)  
      << "dilation must be specified once, or once per spatial dimension "  
      << "(dilation specified " << num_dilation_dims << " times; "  
      << num_spatial_axes_ << " spatial dims).";  
  const int kDefaultDilation = 1;  
  for (int i = 0; i < num_spatial_axes_; ++i) {  
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :  
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);  
  }  
  // Special case: im2col is the identity for 1x1 convolution with stride 1  
  // and no padding, so flag for skipping the buffer and transformation.  
  //判断是不是1*1卷积  
  is_1x1_ = true;  
  for (int i = 0; i < num_spatial_axes_; ++i) {  
    is_1x1_ &=  
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;  
    if (!is_1x1_) { break; }  
  }  
  // Configure output channels and groups.  
  channels_ = bottom[0]->shape(channel_axis_);//获取卷积层输入的单blob的通道数  
  num_output_ = this->layer_param_.convolution_param().num_output();//获取卷积层输出的通道数  
  CHECK_GT(num_output_, 0);//核验输出通道数是否大于零  
  group_ = this->layer_param_.convolution_param().group();//获取卷积组大小  
  CHECK_EQ(channels_ % group_, 0);//核验输入的单blob通道数是否能被卷积组数整除  
  CHECK_EQ(num_output_ % group_, 0)//核验输出通道数是否能被卷积组数整除  
      << "Number of output should be multiples of group.";  
  if (reverse_dimensions()) {//若需要反转卷积操作，则交换输入输出，否则不交换  
    conv_out_channels_ = channels_;  
    conv_in_channels_ = num_output_;  
  } else {  
    conv_out_channels_ = num_output_;  
    conv_in_channels_ = channels_;  
  }  
  // Handle the parameters: weights and biases.  
  // - blobs_[0] holds the filter weights  
  // - blobs_[1] holds the biases (optional)  
  vector<int> weight_shape(2);//定义卷积层参数规格  
  weight_shape[0] = conv_out_channels_;//权重参数shape的第一个数为输出通道大小，即每个输出通道对应各自的卷积核，理解为num  
  weight_shape[1] = conv_in_channels_ / group_;//权重参数shape的第二个数为输入通道大小除以卷积组数，理解为channel  
  for (int i = 0; i < num_spatial_axes_; ++i) {  
    weight_shape.push_back(kernel_shape_data[i]);//权重参数shape的第三个和第四个数为卷积核维度大小  
  }  
  bias_term_ = this->layer_param_.convolution_param().bias_term();//获取是否使用偏置的参数  
  vector<int> bias_shape(bias_term_, num_output_);//定义偏置参数规格，若bias_term_为true(1)，那么bias_shape[0]=num_output_  
  if (this->blobs_.size() > 0) {  
    CHECK_EQ(1 + bias_term_, this->blobs_.size())//核验blobs_是否合法  
        << "Incorrect number of weight blobs.";  
    if (weight_shape != this->blobs_[0]->shape()) {//若weight_shape不为bobs_[0]的shape，则输出异常  
      Blob<Dtype> weight_shaped_blob(weight_shape);  
      LOG(FATAL) << "Incorrect weight shape: expected shape "  
          << weight_shaped_blob.shape_string() << "; instead, shape was "  
          << this->blobs_[0]->shape_string();  
    }  
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {//若bias_shape不为bobs_[1]的shape，则输出异常  
      Blob<Dtype> bias_shaped_blob(bias_shape);  
      LOG(FATAL) << "Incorrect bias shape: expected shape "  
          << bias_shaped_blob.shape_string() << "; instead, shape was "  
          << this->blobs_[1]->shape_string();  
    }  
    LOG(INFO) << "Skipping parameter initialization";  
  } else {//若blobs_.size() = 0，那么根据bias_term_的真伪进行blobs_的大小初始化  
    if (bias_term_) {  
      this->blobs_.resize(2);  
    } else {  
      this->blobs_.resize(1);  
    }  
    // Initialize and fill the weights:  
    // output channels x input channels per-group x kernel height x kernel width  
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));//将blobs_[0]大小初始化为weight_shape  
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(  
        this->layer_param_.convolution_param().weight_filler()));//读取我们定义层的参数中的权重填充，默认为0  
    weight_filler->Fill(this->blobs_[0].get());//进行权重填充  
    // If necessary, initialize and fill the biases.  
    if (bias_term_) {  
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));//若启用了偏置，则读取我们定义层的参数中的偏置填充，默认为0  
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(  
          this->layer_param_.convolution_param().bias_filler()));  
      bias_filler->Fill(this->blobs_[1].get());//进行偏置的填充  
    }  
  }  
  kernel_dim_ = this->blobs_[0]->count(1);//获取一个输出通道对应的所有卷积核对输入的一个卷积组所有通道操作一次处理数据量大小，为(输入总通道数/卷积组数)*卷积核高*卷积核宽  
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;//获取权重的偏移量，理解为(conv_out_channels_/group_)* kernel_dim_   
  // Propagate gradients to the parameters (as directed by backward pass).  
  this->param_propagate_down_.resize(this->blobs_.size(), true);//初始化对权重和偏置(可选)梯度反传的开关  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  const int first_spatial_axis = channel_axis_ + 1;//找到卷积操作处理的第一维的索引，通常为height  
  /*核验输入blob的维度是否等于卷积操作处理的第一维的索引加上卷积操作需要处理的维度数*/  
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)  
      << "bottom num_axes may not change.";  
  num_ = bottom[0]->count(0, channel_axis_);//获取卷积层操作输入的图片数目  
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)//检查输入的通道数是否合法  
      << "Input size incompatible with convolution kernel.";  
  // TODO: generalize to handle inputs of different shapes.  
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {  
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())//如果输入多个blob的话，检查所有blob是否具有相同的shape  
        << "All inputs must have the same shape.";  
  }  
  // Shape the tops.  
  bottom_shape_ = &bottom[0]->shape();//获取卷积层输入的blob的形状  
  compute_output_shape();//获取卷积层输出的blob的形状  
  vector<int> top_shape(bottom[0]->shape().begin(),//初始化top_shape第一个元素为输入单位blob的num  
      bottom[0]->shape().begin() + channel_axis_);  
  top_shape.push_back(num_output_);//top_shape加入输出的通道数  
  for (int i = 0; i < num_spatial_axes_; ++i) {  
    top_shape.push_back(output_shape_[i]);//top_shape加入卷积处理的维度  
  }  
  for (int top_id = 0; top_id < top.size(); ++top_id) {  
    top[top_id]->Reshape(top_shape);//将top的每个blob进行初始化  
  }  
  if (reverse_dimensions()) {  
    /*如果要反转卷积操作，conv_out_spatial_dim_初始化为卷积层输出单位blob(bottom[0])的单通道的数据量*/  
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);  
  } else {  
    /*否则，conv_out_spatial_dim_初始化为卷积层输出单位blob(top[0])的单通道的数据量*/  
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);  
  }  
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;//col_offset表征了一个输出通道对应的所有卷积核处理的一个卷积组的所有数据量  
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;//output_offset_表征了一个卷积组输出的所有数据量  
  // Setup input dimensions (conv_input_shape_).  
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);//用于初始化卷积操作输入数据的形状，一般三维(C,H,W)  
  conv_input_shape_.Reshape(bottom_dim_blob_shape);//初始化卷积层输入shape，一般大小为3  
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();  
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {//初始化卷积层的输入参数，一般顺序为channel->height->width  
    if (reverse_dimensions()) {  
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);  
    } else {  
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);  
    }  
  }  
  // The im2col result buffer will only hold one image at a time to avoid  
  // overly large memory usage. In the special case of 1x1 convolution  
  // it goes lazily unused to save memory.  
  col_buffer_shape_.clear();  
  col_buffer_shape_.push_back(kernel_dim_ * group_);//col_buffer_shape_加入(输入总通道数*卷积核高*卷积核宽)  
  for (int i = 0; i < num_spatial_axes_; ++i) {//col_buffer_shape_加入卷积层输出单通道的维度  
    if (reverse_dimensions()) {  
      col_buffer_shape_.push_back(input_shape(i + 1));  
    } else {  
      col_buffer_shape_.push_back(output_shape_[i]);  
    }  
  }  
  col_buffer_.Reshape(col_buffer_shape_);//初始化col_buffer  
  bottom_dim_ = bottom[0]->count(channel_axis_);//bottom_dim_描述的是bottom blob的一个channel包含的数据量  
  top_dim_ = top[0]->count(channel_axis_);//top_dim_描述的是top blob的一个channel包含的数据量  
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;//描述了一个输出通道对应的所有卷积核对全部输入做卷积操作时转换生成的列向量的数量  
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;//描述了将生成的列向量还原卷积操作的区域图的数量  
  // Set up the all ones "bias multiplier" for adding biases by BLAS  
  out_spatial_dim_ = top[0]->count(first_spatial_axis);//描述了输出的单通道数据量  
  if (bias_term_) {//若启用了偏置，那么初始化偏置乘数blob  
    //偏置乘数的大小为输出的单通道数据量，因为对于每个输出数据乘数不一样  
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);  
    bias_multiplier_.Reshape(bias_multiplier_shape);  
    caffe_set(bias_multiplier_.count(), Dtype(1),//先将这些乘数置为1  
        bias_multiplier_.mutable_cpu_data());  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,//进行数据的cpu前向传播  
    const Dtype* weights, Dtype* output, bool skip_im2col) {  
  const Dtype* col_buff = input;  
  if (!is_1x1_) {  
    if (!skip_im2col) {//im2col将一个卷积操作处理的原特征图按小窗变成并排列向量  
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());  
    }  
    col_buff = col_buffer_.cpu_data();  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /  
        group_, conv_out_spatial_dim_, kernel_dim_,  
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,  
        (Dtype)0., output + output_offset_ * g);  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,//进行偏置的cpu前向传播  
    const Dtype* bias) {  
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,  
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),  
      (Dtype)1., output);  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,//进行数据梯度的cpu反向传播  
    const Dtype* weights, Dtype* input) {  
  Dtype* col_buff = col_buffer_.mutable_cpu_data();  
  if (is_1x1_) {  
    col_buff = input;  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,  
        conv_out_spatial_dim_, conv_out_channels_ / group_,  
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,  
        (Dtype)0., col_buff + col_offset_ * g);  
  }  
  if (!is_1x1_) {  
    conv_col2im_cpu(col_buff, input);//将并列的列向量还原成图像  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,//进行权重的cpu前向传播  
    const Dtype* output, Dtype* weights) {  
  const Dtype* col_buff = input;  
  if (!is_1x1_) {  
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());  
    col_buff = col_buffer_.cpu_data();  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,  
        kernel_dim_, conv_out_spatial_dim_,  
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,  
        (Dtype)1., weights + weight_offset_ * g);  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,//进行偏置梯度的cpu反向传播  
    const Dtype* input) {  
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,  
      input, bias_multiplier_.cpu_data(), 1., bias);  
}  
  
#ifndef CPU_ONLY  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,//进行数据的gpu前向传播  
    const Dtype* weights, Dtype* output, bool skip_im2col) {  
  const Dtype* col_buff = input;  
  if (!is_1x1_) {  
    if (!skip_im2col) {  
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());  
    }  
    col_buff = col_buffer_.gpu_data();  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /  
        group_, conv_out_spatial_dim_, kernel_dim_,  
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,  
        (Dtype)0., output + output_offset_ * g);  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,//进行偏置的gpu前向传播  
    const Dtype* bias) {  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,  
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),  
      (Dtype)1., output);  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,//进行数据梯度的gpu反向传播  
    const Dtype* weights, Dtype* input) {  
  Dtype* col_buff = col_buffer_.mutable_gpu_data();  
  if (is_1x1_) {  
    col_buff = input;  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,  
        conv_out_spatial_dim_, conv_out_channels_ / group_,  
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,  
        (Dtype)0., col_buff + col_offset_ * g);  
  }  
  if (!is_1x1_) {  
    conv_col2im_gpu(col_buff, input);  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,//进行权重的gpu前向传播  
    const Dtype* output, Dtype* weights) {  
  const Dtype* col_buff = input;  
  if (!is_1x1_) {  
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());  
    col_buff = col_buffer_.gpu_data();  
  }  
  for (int g = 0; g < group_; ++g) {  
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,  
        kernel_dim_, conv_out_spatial_dim_,  
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,  
        (Dtype)1., weights + weight_offset_ * g);  
  }  
}  
  
template <typename Dtype>  
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,//进行偏置梯度的反向传播  
    const Dtype* input) {  
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,  
      input, bias_multiplier_.gpu_data(), 1., bias);  
}  
  
#endif  // !CPU_ONLY  
  
INSTANTIATE_CLASS(BaseConvolutionLayer);  
  
}  // namespace caffe  