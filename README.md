# RGB2D_caffe_cpp
Get depth map from one web camera using caffe framework with cpp

## Compiling and usage:

1. Create a folder called cpp_depth/ in ./caffe/examples/, copy depth.cpp to ./caffe/examples/cpp_depth/
2. Copy depth_model/ to ./caffe/models/ (.caffemodel file is too big to upload to github, please download it here: [Link](https://drive.google.com/drive/folders/1TnzjYybQYphd__v5XESVjs9_EOXt6ZWT?usp=sharing))
3. In caffe root directory run:
```
$ cd build/
$ make
$ ./examples/cpp_depth/depth.bin \
  ../models/depth_model/model_norm_abs_100k.prototxt \
  ../models/depth_model/model_norm_abs_100k.caffemodel 
```
If you want it to work on TX1/2, you have to install caffe for TX1: https://github.com/jetsonhacks/installCaffeJTX1 ,
and then do the same thing above.
