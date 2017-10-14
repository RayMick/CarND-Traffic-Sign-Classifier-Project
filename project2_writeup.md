**Udacity Self-driving Car NanoDegree Project2** -- **Traffic Sign Recognition**
---

## **Introduction**

In this project, a deep convolutional neural network is implemented using Python under TensorFlow framework to classify traffic signs. Deep learning techniques and computer vision methods are employed to do the data pre-precosssing and augmentation.  A CNN model is constructed and trained to  classify traffic sign images using the German Traffic Sign Dataset. The model performance is evaluated on hold-out test sets as well as new images found separately online.

---

## **Methodology and data pipeline**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Split the data set into Training/Validation/Testing sub-datasets
* Explore, summarize and visualize the data set
* Pre-process the Data set (normalization, grayscale, etc.)
* Data augmentation using random shift and perspective transformation
* Design, train and test a CNN model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/visualization.png "Visualization"
[image2]: ./writeup_images/color.png "Original Color"
[image3]: ./writeup_images/after-pre-proccessing.png "Random Noise"
[image4]: ./writeup_images/new_image_after_resize.png "Traffic Sign 1"
[image5]: ./writeup_images/new_image_after_grayscale.png "Traffic Sign 1"
[image6]: ./writeup_images/new_image_predictions.png "Traffic Sign 2"
[image7]: ./writeup_images/new_image_probabilities.png "Traffic Sign 3"
[image8]: ./writeup_images/new4.png "Traffic Sign 4"


## **Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
**Writeup / README**

This is the Writeup / README that includes all the rubric points and how each one is addressed. The submission includes the project code.

Please find the source code of the project using the following link [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary and Exploration**

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* <Mark> **The shape of original traffic sign images come in different sizes. THE pickled data are resized to 32x32x3 (the last 3 means it is a 3-channel color image). And later on in the project, these images are converted to 32x32x1 (single channel grayscale images)** </Mark>
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data

![alt text][image1]



**Data Pre-processing and augmentation**

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


Here are examples of a traffic sign image before any preprocessing.

![alt text][image2]

As a first step, I decided to convert the images to grayscale because grayscale suppresses some random noise in the color image also is only single channel image, which reduces the size/dimension of the input images.

Also, I normalized the image data because normalization bring all the input images into the same scale and this allows the downstream models to work on the data that shares the same range of value, allowing a more stable and meaningful numerical calculations.

Lastly, also did a exposure equalization to enhance the contrast of the images before it will highlights the details of the grayscale images.

The results are these preprocessing steps are shown below.


Additional data are generated during the data augmentation step. The reasons are twofold.
First, there are data imbalance in the input images. Certain types of traffic signs are significantly less than some dominating types. If we train the classifier directly with the raw data set, it may be biased towards the classes that have more data.
Secondly, by generating more randomly shifted/modified images the neural network gets to see more data and is suppose to generalize better.

To add more data to the  data set, I used the following techniques.
Image Warping (Affine transformation)
perspective transformation
Random Shift
OpenCV and numpy/scipy are used to implement the above procedures.

Images after augmentation are presented below

![alt text][image3]

Date set after augmentation:
* The size of training set is increased from 34799 to 68274
* The size of the validation set is increased from 4410 to 17069
* The size of test set is kept the same as 12630
* The shape of a traffic sign image is still 32x32
* The number of unique classes/labels in the data set is still 43


**Design and Test a Model Architecture**


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Layer 1 Convolution 5x5     	| shape=(5,5,1,6), 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Layer 2 Convolution 5x5	    | shape=(5,5,6,16), 1x1 stride,  outputs 10x10x16     									|
| RELU					|												|
| Max pooling					|	2x2 stride, 	outputs 5x5x16										|
| Flatten			|	Input 5x5x16=400, 	outputs 120										|
| Layer 3 Fully connected		| Input = 400, Output = 120       									|
| RELU					|												|
| Layer 4 Fully connected		| Input = 120, Output = 84       									|
| RELU					|												|
| Layer 5 Fully connected		| Input = 84, Output = 43       									|
| Softmax				| Outputs probabilities       									|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
* batch size of 64
* number of training epochs is set to 50
* the learning rate is 0.0005
* The neural network weights are randomly generated from a normal distribution with a mean of 0 and sigma of 0.1.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.991
* test set accuracy of 0.946

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The LeNet architecture is chosen as the starting point since it have been proofed working well with traffic sign recognition problems in the literatures.

* What were some problems with the initial architecture?

It is giving a decent accuracy around 89%. I modified it into the implementation as shown in previous table, with 2 convolution layers and 3 fully connected layers.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The architecture is almost kept the same. Only dimensions of the inputs and outputs are modified accordingly to accommodate the size of the input images (32x32).

* Which parameters were tuned? How were they adjusted and why?

Batch size, epochs, leaning rate are all tuned to improve the performance.
The final values are presented in above paragraphs.
I found out that using a relatively small batch size (64) and learning rate (0.0005) help improve the validation accuracy. Also larger number of epochs affects the final prediction performance a lot.
<mark>**During the first submission, I used a total epoch of 50 and it actually leads to overfitting, since the validation accuracy is essentials flat and unchanged after about ~28 iterations. Therefore, for the second submission, I changed the number of epochs from 50 to 28. This prevents the overfitting and the prediction accuracy on the newly images found online also improves a lot! Thanks for the first reviewer's precious feedback!**</mark>

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution layer work works well because the traffic sign images are translational invariant. As long as we find the signs in the image it can be recognized.
Important design choices that affect the final accuracy most is the data pre-precosssing and augmentation. Since it normalizes the images, reduces the noise and class imbalance. These are the key factors that boosts the final validation accuracy to beyond 99%.

**Test a Model on New Images**

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are German traffic signs that I found on the web:

![alt text][image4]

The second image ("pedestrian only") and last image ("no parking") might be difficult to classify because they are not part of the original training data set.

<Mark> **In order to do the recognition of the new images, same pre-precosssing are carried out, namely normalization, exposure equalization and converting to grayscale.** </Mark>


![alt text][image5]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6]

<Mark> **
Except the two new images "pedestrian only", "no parking" and the "speed limit 70km/h", all other images are classified correctly, which gives a total accuracy of 70%.

Close inspection at the three mis-classified images, we can find that even though they are not the right labels, the predictions are not far off. For example, the "pedestrian only" after converting to grayscale indeed looks like "go straight or right", which can be seen from image 5 (the grayscale results for the 10 new images).

There are also quite few possible reasons for this degraded performance compared to the results on the validation and testing data sets (99% and 94%).
**</Mark>


1. The newly download images come in different sizes, a downsampling is performed to resize the images to 32 x 32.
The OpenCV resize function is used to do the downsampling, which generates aliasing. This could affect the correctness of the model since at a low resolution, aliasing could present noisy features that distracts the model.
2. A Gaussian blurring and sharpening operaiton is carried out and the results shows improvements
3. Moreover, the model could be biased or overfitted to the training examples. When presented with these new traffic images (especially the two complete new ones, which are unseen before), the model could not find the correct labels.
4. Adding dropouts could reduce the overfitting and possibly increase the accuracy here.  


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in second last cell of the Ipython notebook.

<Mark> **
The possibilities of each prediction are shown in the image below. We can see that the model is quite certain (100% probability) about the 8 images and these images are classified correctly (with the exception of the "pedestrian only", which again looks overall a lot like the "go straight or right" in grayscale). And even when presented with unseen type of images, such as "no parking", the model is able to find the most resemble existing labels (like "Yield" and "Priority road").
** <Mark>
![alt text][image7]

Again the results may be further improved by adding dropout layers to CNN and also introducing regularization, which could alleviate the overfitting. This is reserved from future studies.  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
