# Planar Mosaic Reconstruction

## Requirements:
* Tested on OSX 10.10.5, Using Python 2.7.10
* OpenCV 3.0.0 (https://github.com/Itseez/opencv/ -- git checkout 3.0.0)
* OpenCV_contrib 3.0.0 (https://github.com/Itseez/opencv_contrib -- git checkout 3.0.0)
* Use CMAKE arguments: cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=OFF -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ ..
* numpy (pip)
* imutils (pip)

## Usage

On a 13" macbook pro, the following command provides a nice balance between screen real estate and processing time.

```
./mosaic.py -d datasets/example1 -is 0.1 -os 0.9
```

``-is`` is the scaling on the input images (smaller value = smaller images, faster processing)
``-os`` is the scaling on the output container (smaller value = smaller output mosaic)
``-m`` is for intermediate matching visualizations.
