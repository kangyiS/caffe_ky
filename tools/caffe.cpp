#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
DEFINE_string(stage, "",
    "Optional; network stages (not to be confused with phase), "
    "separated by ','.");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
             "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
             "Optional; action to take when a SIGHUP signal is received: "
             "snapshot, stop or none.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
//g_brew_map是一个std::map   count(name)是统计在影射中出现了多少次name
  if (g_brew_map.count(name)) {//判断name是不是train,test,device_query,time中的一个
    return g_brew_map[name];//如果是，就调用相应的train(),test(),device_query(),time()
	//返回值的类型是BrewFunction，这是一个函数指针，根据g_brew_map[name]的值，来决定调用train(),test(),device_query(),time()哪个函数
  } else {
    LOG(ERROR) << "Available caffe actions:";//如果不是的话，就打印可以使用的caffe命令
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// Parse GPU ids or use all available devices
//将所有可以使用的GPU设备的编号都保存在gpus里面
static void get_gpus(vector<int>* gpus) {
	//如果要使用所有的GPU
  if (FLAGS_gpu == "all") {
    int count = 0;
#ifndef CPU_ONLY
//检查电脑上一共有多少个GPU，GPU数量保存在count里面
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
//将所有的GPU都记录下来，从0开始，0、1、2 ……
//gpus是一个int向量，里面保存的是gpu的设备编号
    for (int i = 0; i < count; ++i) {
      gpus->push_back(i);
    }
	//如果不使用所有的GPU，那么就检查一下要使用哪些GPU
  } else if (FLAGS_gpu.size()) {
    vector<string> strings;
	//strings里面保存了要使用的GPU的设备编号
    boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
		//lexical_cast()函数可以将字符串转成int数据
      gpus->push_back(boost::lexical_cast<int>(strings[i]));
    }
	//如果FLAGS_gpu里面没有任何数据，那么就检查一下gpus->size()是不是等于零
	//如果不是零，那就奇怪了，应该是在哪里给gpus赋值了
  } else {
    CHECK_EQ(gpus->size(), 0);
  }
}

// Parse phase from flags
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
  if (FLAGS_phase == "")
    return default_value;
  if (FLAGS_phase == "TRAIN")
    return caffe::TRAIN;
  if (FLAGS_phase == "TEST")
    return caffe::TEST;
  LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
  return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
//使用了boost函数库，boost::split()是将FLAGS_stage里面的内容，根据逗号来分割，分割成几个部分，保存在stages里面
vector<string> get_stages_from_flags() {
  vector<string> stages;
  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
  return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
  vector<int> gpus;
  get_gpus(&gpus);
  for (int i = 0; i < gpus.size(); ++i) {
    caffe::Caffe::SetDevice(gpus[i]);
    caffe::Caffe::DeviceQuery();
  }
  return 0;
}
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

// Train / Finetune a model.
int train() {
	//CHECK_GT 和 CHECK 都是Google glog里面的函数，Google glog是一个很好的开源日志函数库
	//GitHub代码：https://github.com/google/glog
	//CHECK_EQ(a,b) 是判断a==b是否成立，如果成立，返回NULL，否则，输出一个字符串
	//CHECK_LE(a,b) 是判断a<=b是否成立，如果成立，返回NULL，否则，输出一个字符串
	//CHECK_LT(a,b) 是判断a< b是否成立，如果成立，返回NULL，否则，输出一个字符串
	//CHECK_GE(a,b) 是判断a>=b是否成立，如果成立，返回NULL，否则，输出一个字符串
	//CHECK_GT(a,b) 是判断a> b是否成立，如果成立，返回NULL，否则，输出一个字符串
	//FLAGS_solver指的是solver.prototxt文件，如果FLAGS_solver.size() == 0，那么说明用户没有传入solver.prototxt文件
	//FLAGS_XXX都在caffe.cpp开头，用DEFINE_XXX定义过，然后执行::gflags::ParseCommandLineFlags()即可得到FLAGS_XXX
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  //这里不允许snapshot和weight同时出现，因为weight是从头训练模型的时候需要的参数，snapshot是接着之前的训练（训练有可能暂停了）继续训练模型时用的参数
  //在这里，由于命令行参数使用的是--weights=“...”，所以FLAGS_weights.size()>0，FLAGS_snapshot.size()=0
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";
/*******************************************************************************/
/*********************这里主要是获得solver_param这个参数的信息******************/
/*******************************************************************************/
//FLAGS_stage应该是空的，在这里能获取stage吗？？？
  vector<string> stages = get_stages_from_flags();

  //SolverParameter定义在caffe.proto里面，是一个message
  //这里使用的是google的protocol buffer代码库，GitHub代码：https://github.com/google/protobuf/
  caffe::SolverParameter solver_param;
  //FLAGS_solver里面保存的是solver.prototxt的文件路径
  //将solver.prototxt文件中的参数，保存到solver_param里面，如果没有FLAGS_solver，程序强行终止
  caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
  //mutable_前缀表示返回对应的指针，set_和add_的含义很明显
  //FLAGS_level = 0, 默认值
  solver_param.mutable_train_state()->set_level(FLAGS_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param.mutable_train_state()->add_stage(stages[i]);
  }
/*******************************************************************************/


/*******************************************************************************/
/**************************这里主要是获得GPU的信息******************************/
/*******************************************************************************/
  // If the gpus flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  /*下面是去查询用户配置的GPU信息，用户可以在输入命令行的时候配置gpu信息，也可以在solver.prototxt 
  文件中定义GPU信息，如果用户在solver.prototxt里面配置了GPU的id，则将该id写入FLAGS_gpu中，如果用户 
  只是说明了使用gpu模式，而没有详细指定使用的gpu的id，则将gpu的id默认为0。*/
  //如果FLAGS_gpu还没有信息，但是却要求使用GPU求解（solver_mode()==caffe::SolverParameter_SolverMode_GPU）
  //那就要给FLAGS_gpu添加信息
  if (FLAGS_gpu.size() == 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
      //如果在solver.prototxt文件里面指定了GPU的设备号，那就添加指定的GPU设备号
	  if (solver_param.has_device_id()) {
          FLAGS_gpu = "" +
              boost::lexical_cast<string>(solver_param.device_id());
	//如果solver.prototxt里面没有指定GPU设备号，那就使用默认的GPU，设备号是0，应该是电脑的主GPU
      } else {  // Set default GPU if unspecified
          FLAGS_gpu = "" + boost::lexical_cast<string>(0);
      }
  }

  vector<int> gpus;
  get_gpus(&gpus);//由于电脑只有一个GPU(GTX1070),所以这里gpus=[0]
  if (gpus.size() == 0) {//如果经历了上面的get_gpus()发现没有GPU的话，就用CPU
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {//如果有GPU，就输出GPU的信息
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i) {
      s << (i ? ", " : "") << gpus[i];
    }
    LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;//这里使用了CUDA库
    for (int i = 0; i < gpus.size(); ++i) {
      cudaGetDeviceProperties(&device_prop, gpus[i]);//获取GPU的设备信息，也就是设备特性
      LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;//将要使用的GPU的设备编号和设备名字打印出来
    }
#endif
    solver_param.set_device_id(gpus[0]);//device_id = gpus[0]
    Caffe::SetDevice(gpus[0]);//让计算机使用gpus[0]
    Caffe::set_mode(Caffe::GPU);//设置mode为GPU，里面用到了一个boost的线程局部存储指针
    Caffe::set_solver_count(gpus.size());//设置gpu数量
  }
/*******************************************************************************/


/*******************************************************************************/
/************************这里主要是创建solver并开始训练*************************/
/*******************************************************************************/
//检查控制台信号，允许有stop和snapshot，在下面的solve的过程中，可以用ctrl-c来退出训练
//FLAGS_sigint_effect = 'stop', FLAGS_sighup_effect = 'snapshot'
//FLAGS_sigint_effect和FLAGS_sighup_effect都是在运行程序之前，通过控制台传入的参数
//在这里控制台没有传入这两个参数，所以就是默认值'stop'和'snapshot'
  caffe::SignalHandler signal_handler(
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));

  /*下面就开始构造网络训练器solver，调用SolverRegistry的CreateSolver函数得到一个solver，在初始化solver的过程中， 
  使用了之前解析好的用户定义的solver.prototxt文件，solver负担了整个网络的训练责任，假设使用SGD方法：
  创建的过程中，依次调用了net_(), callbacks_(), root_solver_(root_solver), requested_early_exit_(false), 
  Solver<Dtype>::Init(param), SGDSolver<Dtype>::PreSolve()，在这一连串的方法中，
  比较重要的是Solver<Dtype>::Init(param)，在这个方法中，使用train.prototxt初始化了训练网络和测试网络
  solver这个智能指针，最终指向了SGDSolver<Dtype>(param)这个对象，这个对象是用new的方式来创建的*/
  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());

  if (FLAGS_snapshot.size()) {//如果有snapshot，说明用户是要接着以前的训练继续训练
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  } else if (FLAGS_weights.size()) {//如果有weights，说明用户是要开始一个新的训练，我的程序里用的是weights
  //FLAGS_weights里面保存的是.caffemodel的文件路径，这是通过命令行参数传入的
    CopyLayers(solver.get(), FLAGS_weights);
  }

  if (gpus.size() > 1) {//如果GPU数量不止1个，则开启多GPU训练模式
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  } else {//如果只有1个GPU，那么就开始训练了，进入Solve()
    LOG(INFO) << "Starting Optimization";
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
/*******************************************************************************/
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
  caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
  vector<string> stages = get_stages_from_flags();

  // Set device id and mode
  vector<int> gpus;
  get_gpus(&gpus);
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  const vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  const vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  Timer backward_timer;
  Timer timer;
  std::vector<double> forward_time_per_layer(layers.size(), 0.0);
  std::vector<double> backward_time_per_layer(layers.size(), 0.0);
  double forward_time = 0.0;
  double backward_time = 0.0;
  for (int j = 0; j < FLAGS_iterations; ++j) {
    Timer iter_timer;
    iter_timer.Start();
    forward_timer.Start();
    for (int i = 0; i < layers.size(); ++i) {
      timer.Start();
      layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
      forward_time_per_layer[i] += timer.MicroSeconds();
    }
    forward_time += forward_timer.MicroSeconds();
    backward_timer.Start();
    for (int i = layers.size() - 1; i >= 0; --i) {
      timer.Start();
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          bottom_vecs[i]);
      backward_time_per_layer[i] += timer.MicroSeconds();
    }
    backward_time += backward_timer.MicroSeconds();
    LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
      << iter_timer.MilliSeconds() << " ms.";
  }
  LOG(INFO) << "Average time per layer: ";
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
      "\tforward: " << forward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername  <<
      "\tbackward: " << backward_time_per_layer[i] / 1000 /
      FLAGS_iterations << " ms.";
  }
  total_timer.Stop();
  LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
    FLAGS_iterations << " ms.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

/*通过下面这段命令，来执行程序
cd /home/ubuntu/caffe
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300_score/solver.prototxt" \
--weights="models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300_score/VGG_VOC0712_SSD_300x300_test120000.log
*/
int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;//日志记录在文件中，同时也打印到控制台中
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));//设置版本信息，CAFFE_VERSION是在Makefile中定义的
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"      //设置命令行帮助信息，当命令行参数错误或加-help选项时，可以打印帮助信息
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
    try {
#endif
      //argv[1] = train，这个是由ssd_pascal.py最后一部分代码配置的（job_file）
	  //根据命令行，来确定调用了哪个函数，train(), test(), device_query() 或是 time()
      return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
    } catch (bp::error_already_set) {
      PyErr_Print();
      return 1;
    }
#endif
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
