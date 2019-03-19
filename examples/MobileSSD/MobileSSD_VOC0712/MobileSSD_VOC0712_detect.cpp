/*
 * compute speed without storing result,using VOC2007 dataset: 
 * using depthwise convolution:
 * video:      180fps(forward), 154fps(loop)
 * one image:  90fps(forward),  71fps(loop) 
 * images:     220fps(forward), 153fps(loop)
 * 
 */
#include <caffe/caffe.hpp>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace caffe;
using namespace std;
using namespace cv;
using google::protobuf::io::FileInputStream; 
using google::protobuf::Message; 

#define SAVE_IMAGE true

string network_name = "MobileSSD";
string dataset_name = "VOC0712";
string model_name = network_name + "_" + dataset_name;
string job_dir = "examples/" + network_name + "/" + model_name;
string prototxt_dir = job_dir + "/prototxt";
string trainLog_dir = job_dir + "/log";
string trainData_dir = job_dir + "/data";
string trainModel_dir = job_dir + "/model";
const char* source_file = "/home/kangyi/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg";
//const char* source_file = "/home/kangyi/data/VOCdevkit/VOC2007/JPEGImages";
//const char* source_file = "/home/kangyi/caffe/examples/videos/ILSVRC2015_train_00755001.mp4";
string model_file = trainModel_dir + "/MobileNetSSD7000_deploy.caffemodel";
string deploy_file = prototxt_dir + "/MobileNetSSD_deploy_dw.prototxt";
string labelmap_file = trainData_dir + "/labelmap_voc.prototxt";
string result_dir = job_dir + "/result/detect";
string dataset_mean = "104,117,123";//trainData_dir + "/VOC0712_trainval_mean.binaryproto";//"104,117,123";
string file_type = "image"; // "image" "images" or "video"

vector<string> labelmap;

double d_time=0, s_time=0, f_time=0;

class Detector
{
    public:
    Detector(const string& model_file,
             const string& deploy_file,
             const string& dataset_mean);
    std::vector<vector<float> > Detect(const cv::Mat& img);
    
    private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    void SetMean(const string& dataset_mean);

    private:
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& deploy_file,
                   const string& dataset_mean)
{
    Caffe::set_mode(Caffe::GPU);
    /* Load the network. */
    net_.reset(new Net<float>(deploy_file, TEST));
    net_->CopyTrainedLayersFrom(model_file);
    
    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    SetMean(dataset_mean);  

}
/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& dataset_mean) 
{
    cv::Scalar channel_mean;
    if (dataset_mean.find(",") != string::npos) // mean value
    {
        stringstream ss(dataset_mean);
        vector<float> values;
        string item;
        while (getline(ss, item, ','))
        {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
          "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) 
        {
            /* Extract an individual channel. */
            cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
    else if (dataset_mean.find(".") != string::npos) //mean file
    {
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(dataset_mean.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
          << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) 
        {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
         * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
    }
    else
    {
        LOG(FATAL) << "Wrong dataset mean format!";
    }
}
/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) 
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) 
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Detector::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) 
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    // This operation will write the separate BGR planes directly to the
    // input layer of the network because it is wrapped by the cv::Mat
    // objects in input_channels. 
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
          << "Input channels are not wrapping the input layer of the network.";
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) 
{
	Timer fwd_timer;
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
    // Forward dimension change to all layers. 
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    fwd_timer.Start();
    net_->Forward();
	f_time += fwd_timer.MilliSeconds();

    // Copy the output layer to a std::vector 
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    const int num_det = result_blob->height();
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) 
    {
        if (result[0] == -1)
        {
        // Skip invalid detection.
            result += 7;
            continue;
        }
        vector<float> detection(result, result + 7);
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

vector<string> parseLabelmap(const string& labelmap_file)
{
    LabelMap label_list;
    vector<string> label;
    caffe::ReadProtoFromTextFile(labelmap_file, &label_list);
    label.push_back("background");
    for (int i=1; i<label_list.item_size(); i++)
    {
        label.push_back(label_list.item(i).name());
    }
    return label;
}

string find_basename(const string& filename)
{
    int pos = filename.find_last_of('/');
    string str(filename.substr(pos+1));
    return str;
}

vector<string> getFiles_from_dir(const char* path)
{
	vector<string> files;
	struct dirent* filename;
	DIR* dir;
	dir = opendir(path);
	if (dir == NULL)
	{
		cout << "Can not open " << path << endl;
		vector<string> r(0);
		return r;
	}
	while ((filename = readdir(dir)) != NULL)
	{
		if(strcmp(filename->d_name, ".") ==0 || 
           strcmp(filename->d_name, "..") ==0)
			continue;
		files.push_back((string)path+"/"+filename->d_name);
	}
	return files;
}


int main(int argc, char** argv) 
{
    ::google::InitGoogleLogging(argv[0]);
    
    // parse labelmap file
    labelmap = parseLabelmap(labelmap_file);

    // init detector
    Detector detector(model_file, deploy_file, dataset_mean);

	#if (SAVE_IMAGE == true)
	// Set the output mode.
	std::streambuf* buf = std::cout.rdbuf();
	std::ofstream outfile;
	string out_file = result_dir + "/result.txt";
	if (!out_file.empty()) 
	{
		outfile.open(out_file.c_str());
		if (outfile.good()) 
			buf = outfile.rdbuf();
	}
	std::ostream out(buf);
	#endif

	Timer det_timer;
	if (file_type == "image")
	{
		// read image file and detect
		det_timer.Start();
		cv::Mat img = cv::imread(source_file,-1);
		CHECK(!img.empty()) << "Unable to decode image " << source_file;
		std::vector<vector<float> > detections = detector.Detect(img);
		d_time = det_timer.MilliSeconds();
		
		cout << "forward time: " << f_time << " ms" << endl;
		cout << "detection time: " << d_time << " ms" << endl;

		// save results
		#if (SAVE_IMAGE == true)
		det_timer.Start();
		out << "file_name: " << source_file << std::endl;
		out << "param_name: xmin ymin xmax ymax label_id confidence label_name" << std::endl;
		for(int i=0; i<detections.size(); i++)
		{
			const vector<float>& d=detections[i];
			CHECK_EQ(d.size(),7);
			if(d[2]>0.5)
			{
			    int label_id = static_cast<float>(d[1]);
			    float score = static_cast<float>(d[2]);
			    int xmin = static_cast<int>(d[3] * img.cols);
			    int ymin = static_cast<int>(d[4] * img.rows);
			    int xmax = static_cast<int>(d[5] * img.cols);
			    int ymax = static_cast<int>(d[6] * img.rows);
			    string label_name = labelmap[d[1]];

			    out << "object: ";
			    out << xmin << " "; 
			    out << ymin << " ";
			    out << xmax << " ";
			    out << ymax << " ";
			    out << label_id << " "; 
			    out << score << " ";
			    out << label_name << std::endl;
			    cv::rectangle(img,cvPoint(xmin,ymin),cvPoint(xmax,ymax),cv::Scalar(255,0,0),5);
			    stringstream ss;
			    ss << score << flush;
			    string text = label_name + " " + ss.str();
			    cv::putText(img,text,cvPoint(xmin,ymin+20),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,0,255));
			}
		}
		out << "---" << std::endl;

		string basename;
		basename = find_basename(source_file);
		cv::imwrite(result_dir+"/"+basename, img);
		s_time = det_timer.MilliSeconds();
		cout << "store time: " << s_time << " ms" << endl;
		#endif
	}
	else if (file_type == "images")
	{
		vector<string> files;
		files = getFiles_from_dir(source_file);
		
		unsigned long int img_count=0;
		for (;img_count<files.size();img_count++)
		{
			// read image file and detect
			det_timer.Start();
			cv::Mat img = cv::imread(files[img_count],-1);
			CHECK(!img.empty()) << "Unable to decode image " << files[img_count];
			std::vector<vector<float> > detections = detector.Detect(img);
			d_time += det_timer.MilliSeconds();
			if ((img_count+1)%1000 == 0)
			{
				cout << "Processed " << (img_count+1) << " images... "
                     << "average forward time: " 
                     << f_time/1000.0 << " ms per image" << endl;
				f_time = 0;
				cout << "Processed " << (img_count+1) << " images... "
                     << "average detection time: " 
                     << d_time/1000.0 << " ms per image" << endl;
				d_time = 0;
			}

			// save results
			#if (SAVE_IMAGE == true)
			det_timer.Start();
			out << "file_name: " << files[img_count] << std::endl;
			out << "param_name: xmin ymin xmax ymax label_id confidence label_name" << std::endl;
			for(int i=0; i<detections.size(); i++)
			{
				const vector<float>& d=detections[i];
				CHECK_EQ(d.size(),7);
				if(d[2]>0.5)
				{
					int label_id = static_cast<float>(d[1]);
					float score = static_cast<float>(d[2]);
					int xmin = static_cast<int>(d[3] * img.cols);
					int ymin = static_cast<int>(d[4] * img.rows);
					int xmax = static_cast<int>(d[5] * img.cols);
					int ymax = static_cast<int>(d[6] * img.rows);
					string label_name = labelmap[d[1]];

					out << "object: ";
					out << xmin << " "; 
					out << ymin << " ";
					out << xmax << " ";
					out << ymax << " ";
					out << label_id << " "; 
					out << score << " ";
					out << label_name << std::endl;
					cv::rectangle(img,cvPoint(xmin,ymin),cvPoint(xmax,ymax),cv::Scalar(255,0,0),5);
					stringstream ss;
					ss << score << flush;
					string text = label_name + " " + ss.str();
					cv::putText(img,text,cvPoint(xmin,ymin+20),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,0,255));
				}
			}
			out << "---" << std::endl;

			string basename;
			basename = find_basename(files[img_count]);
			cv::imwrite(result_dir+"/"+basename, img);
			s_time += det_timer.MilliSeconds();
			if ((img_count+1)%1000 == 0)
			{
				cout << "Processed " << (img_count+1) << " images... "
                     << "average store time: " 
                     << s_time/1000.0 << " ms per image" << endl;
				s_time = 0;
			}
			#endif
		}
		if (img_count%1000 != 0)
		{
			cout << "Processed " << img_count << " images... "
                     << "average forward time: " 
                     << f_time/(img_count%1000) << " ms" << endl;
			f_time = 0;
			cout << "Processed " << img_count << " images... "
                     << "average detection time: " 
                     << d_time/(img_count%1000) << " ms" << endl;
			d_time = 0;
			#if (SAVE_IMAGE == true)
			cout << "Processed " << img_count << " images... "
                     << "average store time: " 
                     << s_time/(img_count%1000) << " ms" << endl;
			s_time = 0;
			#endif
		}
	}
	else if (file_type == "video")
	{
		cv::VideoCapture cap;
		cap.open(source_file);
		if(!cap.isOpened())
		{
			cout << "Can not open the video file!" << endl;
			return 0;
		}
		cv::Mat img;
		unsigned long int frame_cnt = 0;

		#if (SAVE_IMAGE == true)
		VideoWriter v_write;
		string result_file = result_dir + "/result.avi";
		cv::Size size = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		v_write.open(result_file,CV_FOURCC('M', 'J', 'P', 'G'), 10, size, true);
		#endif

		while(1)
		{
			cap>>img;
			if(img.empty())
			{
				cout << "Reading video is finished!" << endl;
				break;			
			}
			det_timer.Start();
			std::vector<vector<float> > detections = detector.Detect(img);
			d_time = det_timer.MilliSeconds();
			frame_cnt++;			
			
			cout << "Processed " << frame_cnt << " frames *********************" << endl;
			cout << "forward time: " << f_time << " ms" << endl;
			cout << "detection time: " << d_time << " ms" << endl;
			f_time = 0;

			// save results
			#if (SAVE_IMAGE == true)
			det_timer.Start();
			for(int i=0; i<detections.size(); i++)
			{
				const vector<float>& d=detections[i];
				CHECK_EQ(d.size(),7);
				if(d[2]>0.5)
				{
					int label_id = static_cast<float>(d[1]);
					float score = static_cast<float>(d[2]);
					int xmin = static_cast<int>(d[3] * img.cols);
					int ymin = static_cast<int>(d[4] * img.rows);
					int xmax = static_cast<int>(d[5] * img.cols);
					int ymax = static_cast<int>(d[6] * img.rows);
					string label_name = labelmap[d[1]];

					cv::rectangle(img,cvPoint(xmin,ymin),cvPoint(xmax,ymax),cv::Scalar(255,0,0),5);
					stringstream ss;
					ss << score << flush;
					string text = label_name + " " + ss.str();
					cv::putText(img,text,cvPoint(xmin,ymin+20),cv::FONT_HERSHEY_COMPLEX,1,cv::Scalar(0,0,255));
				}
			}
			v_write.write(img);
			s_time = det_timer.MilliSeconds();
			cout << "store time: " << s_time << " ms" << endl;
			#endif
		}
		cap.release();
	}
	else
	{
		cout << "Incorrect file type!" << endl;
		cout << "You are supposed to choose from 'image', 'images' or 'video'!" << endl;
	}
    return 0;
}
