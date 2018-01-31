
# RGB2D_caffe_cpp
Get depth map from one web camera using caffe framework with cpp and using it to create 3D and SLAM.

## Compiling and usage:

1. Create a folder called cpp_depth/ in ./caffe/examples/, copy all the files to ./caffe/examples/cpp_depth/
2. Create a folder called depth_model/ in ./caffe/models/ and copy .caffemodel and .protxt file into it(.caffemodel file is too big to upload to github, please download it here: [Link](https://drive.google.com/drive/folders/1TnzjYybQYphd__v5XESVjs9_EOXt6ZWT?usp=sharing)).
3. Build the caffe again:
```
$ cd build/
$ cmake ..
$ make
```
4. If you have a camera and need to run in real time, connect your webcamera to the PC and in caffe root directory run(now in caffe/build/):
```
$ ./examples/cpp_depth/depth_camera.bin \
  ../models/depth_model/model_norm_abs_100k.prototxt \
  ../models/depth_model/model_norm_abs_100k.caffemodel 
```
or you can also using directly the python script(now in caffe/):
```
python ./examples/cpp_depth/depth_camera.py
```
5. If you have a folder of RGB images and need to create depth images from that(change the input and output folder path in the script):
```
cd ..
python ./examples/cpp_depth/depth_images.py
```
or you can also using the cpp script (not done yet, update later):
```

```

## Run on TX1/2
If you want it to work on TX1/2, you have to install caffe for TX1: https://github.com/jetsonhacks/installCaffeJTX1,
and then do the same thing above.


## Create 3D and SLAM from RGB and Depth images

1. Make and install [[RTABmap](https://github.com/introlab/rtabmap)].
2. Put the rgb images in <IMAGEFOLDER>/rgb_sync, the depth images in <IMAGEFOLDER>/depth_sync, and run to create rtabmap.db and rtbmap_poses.txt:
```
 ./rtabmap-rgbd_dataset        --Vis/EstimationType 1       --Vis/BundleAdjustment 1       --Vis/PnPReprojError 1.5       --Odom/GuessMotion true       --OdomF2M/BundleAdjustment 1       --Rtabmap/CreateIntermediateNodes true       --Rtabmap/DetectionRate     <IMAGEFOLDER>
```
Example for the TUM data:
```
 ./rtabmap-rgbd_dataset        --Vis/EstimationType 1       --Vis/BundleAdjustment 1       --Vis/PnPReprojError 1.5       --Odom/GuessMotion true       --OdomF2M/BundleAdjustment 1       --Rtabmap/CreateIntermediateNodes true       --Rtabmap/DetectionRate     /media/enroutelab/sdd/data/tum_depth/rgbd_dataset_freiburg3_long_office_household
```
3. Run ./rtabmap -> Tools -> Edit database -> Load rtabmap.db created.
4. Edit -> Regenerate local grid maps
5. View -> 3D View


