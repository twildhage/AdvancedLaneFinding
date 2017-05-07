## Writeup Report **Advanced Lane Finding Project**
---
The goal of this project is to find the highway road lines from a video stream and calculate the curvature radius of the lane and the vehicle position relative to the lane center.
The steps applied to achieve these goals are the following:

* **Camera Calibration:** Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* **Distortion Correction:** Apply a distortion correction to the raw video frames.
* **Binary Line Image:** Use color transforms, gradients, etc., to create a thresholded binary image.
* **"Birds-Eye View":** Apply a perspective transform to rectify binary image.
* **Lane Detection:** Detect lane pixels and fit to find the lane boundary.
* **Lane Curvature and Vehicle Position:** Determine the curvature of the lane and vehicle position with respect to center.
* **Camera View:** Warp the detected lane boundaries back onto the original image in camera view.
* **Decorate Image:** Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./results/calibration_analysis.png "Analysis of Calibrated Images"
[image02]: ./results/calibration_matrix_analysis.png "Analysis of Calibration Matrix"
[image10]: ./results/binary_original_and_birdseye_view.png
"Binary Image in Original and Birdseye View"

[image20]: ./results/lanes_birdseye_view.png
"Image of Detected Lines and Search Region"
[equation1]: ./results/eq1.png "Equation for Curvature Radius"
[video1]: ./project_video.mp4 "Video"

## Project Files and Folders
The following files contain the source code of this project:
* `notebook.ipynb` Main project file. From here all the code is executed.
* `pipeline.py`  All pipeline functions
* `Lane` The Lane class contains line data and performs the line detection.
---


### Camera Calibration

In order to calibrate the camera, the 17 provided chessboard images are used.
In order to validate the quality of the calibration, the camera matrices and the distortion coefficients are evaluated.
For the camera matrix the first singular value is used for validation.

Plotting the distortion coefficients and the first singular value of the camera matrix, one can observe that in both case, the plots show a converging behavior.
I interpret this as a sign, that the calibration became "stationary" and more samples would not further increase the quality of the calibration.
![alt text][image02]
The following picture show an original and an undistorted image in the first row and the overlay and difference of the two in the second row.
In the difference image it can be observed that the distortion increases towards the edges of the image an is nearly zero in the center.
![alt text][image01]
Hence, as a first step of the pipeline, each image of the video stream is undistorted.

### Binary Line Image
In order to identify the lines within the image, several filter techniques are applied.
The basic idea is to scale the image and then threshold a certain representation of the image.
As one example one could convert the image from the RGB colorspace the the HLS colorspace, select the S channel and filter all pixels within a certain range.
The thresholds have be tuned as follows. I started for each representation with the maximum range ([0, 255] or [0, 1] depending on scaling) and increased the lower threshold step wise until the lines started to disappear. I then repeated with reducing the upper threshold in the same way.
This technique is applied to the following image representations:
* x- and y-gradient
* x,y magnitude of the gradient
* direction
* R,S channels

All binary images are then combined.
Here is the code that performs the final combination:
```python
binary_combined[((binary_gradx == 1) & (binary_grady == 1)) |
             ((binary_mag_gradxy == 1) & (binary_dir == 1)) |
                   ((binary_r == 1) &  (binary_s ==1))] = 1
```
### Birds-Eye View
The combined binary image from the last section is then transformed to a birds-Eye view.
The source rectangular is determined via a straight-view picture from the vehicles camera. The vanishing point be found by hand as the crossing point of all lines at the horizon.
The destination rectangular is chosen such that the resulting representation allows curved lanes to be fit on the image.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 225, 700      | 325, 720       |
| 595, 450      | 325, -100      |
| 690, 450      | 955, -100      |
| 1055, 700     | 955, 720       |

![alt text][image10]
As can be seen from the image above, are the lines approximately 700 pixels apart. This is aligned with the default conversion parameter from pixels to meters that is given in the lecture.

### Lane Detection
The binary birds-eye view pictures from the previous section are the inputs for the lane detection.
In order to find the lines in the binary image, first a window search is performed. This is based on the lectures histogram peak detection to locate the starting points of the lines.
Once the lines are found, the search algorithm is switched from 'window' to 'ancestor' search. The latter searches only in vicinity of the previous lines.
Having selected a set of pixels for the left and right line, and polynomial of 2. order is fit to the data.
The current lines and the parameter from the curve fitting step are stored for the moving average filter.
The moving average filter helps to reduce the fluctuations of the results. I chose the number of averaging steps to fit the frame rate. The rational behind this is, to smooth the data on the time scale of one second, which should be enough for cars traveling with a speed of 120 km/h on a highway.
The result of this step can be seen in the following image.
![alt text][image20]
### Radius and Vehicle Position
The radius is calculated based on the average fit parameters. It is calculated according to the formula given in the lecture.
![alt text][equation1]
This equation becomes singular when A goes to zero. This happens when the curvature goes to zero, hence for straight lines. Therefore the displayed radius saturates at 10000 meters.

The vehicle position is calculated by taking the distance between the lines, subtracting the center pixel and deviding the result by two:
```python
self.vehicle_position = ((xlane['right']  - xlane['left']) - xcenter)*self.xmeter_per_pixel
```
This way negative (positive) values correspond to the vehicle being left (right) from the center of the lane.

----------------

### Project Video

Here's a [link to my video result](./project_video_processed.mp4)
[Video on Youtube](https://www.youtube.com/watch?v=LP39pNEHQD4)

---

### Discussion

This project was quite challenging due to the many steps of the pipeline and how they interelate. Getting the warped image correct took me a lot of try and error. Once this part was done, the rest worked far better.

It is great to have a working lane detection, however the implementation will likely fail under less constraint/ideal circumstances.
Here is a list of scenarios and cases which might cause the pipeline to fail in the real world.

* rain, snow, fog
* dense traffic in front
* missing lines
* distracting lines (e.g. building site)
* sharp curves -> one line out of view
* steep streets -> horizon shifts out/down
* local areas (passenger, bicyles, parking cars, ...)

#### Ideas
* Use image recognition to determine what are lines. Here one could prepare a set of real images and ideal binary line images as labels and train the model to output the lines as binary image.
* Use image recognition to determine where it is save to drive when no lines are available. It might be possible to take computer game engines, where to roads and lines are objects and can be marked. This way one should be able to create large training sets.
* Combine behaviour cloning to validate the steering angles (which are derived from velocity, position, lane radius, ...)
*  Use model based tools (e.g. Stateflow) to build a state machine which, among other things, validates the current lines and defines reactions when it is no longer save to drive. Test cases have to be set up and be passed.
