#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using std::string;

class Classifier {
 public:
  Classifier(const string& model_file, const string& trained_file);
  cv::Mat Predict(const cv::Mat& img);

 private:
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
 
  //Blob<float>* output_layer = net_->output_blobs()[0];
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);//data saved for each channel
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
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
/*
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);
*/
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);
  /* check the copy is correct */
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

cv::Mat Classifier::Predict(const cv::Mat& img) {

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  std::cout << "Input channels:" << num_channels_
            << ",Input height:" << input_layer->height()
            << ",Input width:" << input_layer->width() << std::endl;

  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);//bound input_channels to input of net_
  Preprocess(img, &input_channels); //preporcess the image into input_channels;

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  std::cout << "Output Channels:" << output_layer->channels() 
            << ",Output height:" << output_layer->height()
            << ",Output width:" << output_layer->width() << std::endl;

  int width = output_layer->width();
  int height = output_layer->height();
  float* output_data = output_layer->mutable_cpu_data();

  cv::Mat mat(height, width, CV_32FC1, output_data);

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  return mat;
}

int main(int argc, char** argv) {

  string model_file   = argv[1];
  string trained_file = argv[2];
  
  Classifier classifier(model_file, trained_file);//ini

/*
  string file = argv[3];
  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;
  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;

  Mat depth_mat = classifier.Predict(img);//run net
  cv::imwrite("depth_result.png", mat, compression_params);
*/

  //Open camera
  VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;

  for(;;)
  {
    Mat frame;
    cap >> frame; // get a new frame from camera
    Mat depth = classifier.Predict(frame); //run the net, return CV_32FC1

    cv::log(depth,depth);
    depth = 2 * 0.179581 * depth + 1;
    //depth = np.clip(depth, 0.001, 1000)	
    //return np.clip(2 * 0.179581 * np.log(depth) + 1, 0, 1)


    //convert the frame from CV_8C3 to CV_32FC3
    Mat frame_convert;
    frame.convertTo(frame_convert, CV_32FC3); // this is just a copy
    frame_convert /= 255; //divided bye 0xff, it can be showed;

    //change the depth from CV_32FC1 to CV_32FC3 and then resize;
    Mat depth_convert;
    cv::cvtColor(depth, depth_convert, cv::COLOR_GRAY2BGR);
    resize(depth_convert, depth_convert, Size(frame.cols,frame.rows), 0, 0, INTER_CUBIC); 
    
    //to show the two images side by side
    Mat win_mat(cv::Size(2*frame.cols, frame.rows), CV_32FC3);
    frame_convert.copyTo(win_mat(cv::Rect(  0, 0, frame.cols, frame.rows)));
    depth_convert.copyTo(win_mat(cv::Rect(frame.cols, 0, frame.cols, frame.rows)));
    imshow("Images", win_mat);

    if(waitKey(30) >= 0) break;
  }
}
