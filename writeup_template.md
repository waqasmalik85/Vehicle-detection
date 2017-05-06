##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The steps of this project are the following:

* Vehicle and non-vehicle data anylysis
* Anylysis of features Histogram of Oriented Gradients (HOG), binned color features and histogram of colors in different
* Normalization of features
* Training and application of a linear Support Vector Machine(SVM) to classify the images between cars and non cars
* Performance anylysis on different color spaces RGB, YUV, YCrCb
* Implementation of a sliding window to extract features on from a video stream
* Buffer the vehicle candidates over last 10 measurements and created heatmaps
* Use thresholding to avoid false possitives and estimated the bounding box
* Created the final video project_output.mp4

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/carvsno_car.jpg
[image9]: ./output_images/car_HOG_Y.jpg
[image10]: ./output_images/nocar_HOG_Y.jpg
[image11]: ./output_images/car_HOG_Cr.jpg
[image12]: ./output_images/nocar_HOG_Cr.jpg
[image13]: ./output_images/car_HOG_Cb.jpg
[image14]: ./output_images/nocar_HOG_Cb.jpg
[image15]: ./output_images/car_spatial.jpg
[image16]: ./output_images/nocar_spatial.jpg
[image17]: ./output_images/car_Chist.jpg
[image18]: ./output_images/nocar_Chist.jpg
[image19]: ./output_images/car_HOG.jpg
[image20]: ./output_images/nocar_HOG1.jpg
[image21]: ./output_images/all_boxes.jpg
[image22]: ./output_images/detections.jpg
[image23]: ./output_images/detections2.jpg
[image24]: ./output_images/final4.jpg
[image25]: ./output_images/final5.jpg
[image26]: ./output_images/final6.jpg
[image27]: ./output_images/final2.jpg
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  



###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fouth and fifth code cell of the Jupyter notebook in the function `data_look`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  There are 8792 and 8968 vehicles and nonvehicles examples repectively. Each image is 64x64x3. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image8]

I then explored  `RGB`, `YUV` and `YCrCb` color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

####2. Explain how you settled on your final choice of HOG parameters.

After experienting with `RGB`, `YUV` and `YCrCb` color spaces, `YCrCb` showed best results. In `YCrCb` space least false positives have been observed.

I notices that by reducing the size of the image to 16x16 for the bin_spatial feature performance remains more or less but the feature vector size reduces significantly. Below is an example of the feature:-

![alt text][image15]
![alt text][image16]

Color histograms were also included in the feature vector with 16 bins longs histogram. Following is an example of color_hist feature:-

![alt text][image17]
![alt text][image18]

For HOG feature I used all channels, following is my final choice of features:-


*  orient = 9  
*  pix_per_cell = 8
*  cell_per_block = 2
*  hog_channel = 'ALL'
*  spatial_size = (16, 16)
*  hist_bins = 16    



Images below show HOG features extracted by the parameters mentioned above:-

![alt text][image19]
![alt text][image20]

Code cell contains the function `extract_features()` which returns the complete feature vector. Length of the final feature vector is 6108.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Code cell 2 contains the feature extraction from the training data, splitting into train and validation data. Feature normalization and finally training a linear Support Vector Machine(SVM) claassifier training. Features are good enough to separate the data linearly so there was no need to use Kernel trick. Linear SVM with a feature vector of length 6108 showed an accuracy of 99.01%.

It took 17.66 seconds to train the classifier on the laptop.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the code cell 7 function `find_cars()` a single scale search is performed using `64x64` window size and 75% overlap. Original image is caled down to 1.5 times. Thus the window size in original image is `96x96`. A total of 294 windows are searched. Following image shows all the windows displayed on the original image:-

![alt text][image21]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on a single scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are two example images of the images with windows classified as containing cars:-

![alt text][image22]
![alt text][image23]
---

### Video Implementation

####1. Provide a link to your final video output.  

Here is the link to youtube video Video available at youtube [Youtube Link](https://www.youtube.com/watch?v=NSM3LkZe73c)
Link to the repository is at the next link [ video result](./project_output.mp4).

Around 27 seconds white car is lost for approximately 2 seconds and appears again. It is uncretical because the car is neither in the ego lane not changing the lane towards host lane.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In code cell 10, all the boxes with hits over time are kept in a variable. Heat map is created using `add_heat()` last 10 measurements over time. Thus giving the pixels with multiple hits more heat which is directly proportional to the probability of existance of a car. A threshold `apply_threshold()` of 5 hits is used on the heatmap to reduce the number of false positives.   I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Following are some examples of detected boxes, corresponding heatmaps and labels:-

![alt text][image24]
![alt text][image26]
![alt text][image25]





### Following is an example of the final bouning box:

![alt text][image27]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* One of the major issue is the processing time(1.75fps) and computional complexity. It is very importan to reduce the processing time to run the algorithm in real time.
* Robustness can be brought by using filtering techniques like Kalman filter, however host dynamics like speed and yaw rate are unavailable which means the kalman predictions could deviate from the measurements a lot.
* Cameras are unlikely to produce good results during night time. Algorithm may not work at all during night, bad weather, tunnels etc
