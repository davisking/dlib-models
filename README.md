# dlib-models
This repository contains trained models created by me (Davis King).  They are provided as part of the dlib example programs, which are intended to be educational documents that explain how to use various parts of the dlib library.  As far as I am concerned, anyone can do whatever they want with these model files as I've released them into the public domain.  Details describing how each model was created are summarized below. 

* dlib_face_recognition_resnet_model_v1.dat.bz2
  
  This model is a ResNet network with 27 conv layers.  It's essentially a version of the ResNet-34 network from the paper Deep Residual Learning for Image Recognition by He, Zhang, Ren, and Sun with a few layers removed and the number of filters per layer reduced by half.  

  The network was trained from scratch on a dataset of about 3 million faces. This dataset is derived from a number of datasets.  The face scrub dataset (http://vintage.winklerbros.net/facescrub.html), the VGG dataset (http://www.robots.ox.ac.uk/~vgg/data/vgg_face/), and then a large number of images I scraped from the internet.  I tried as best I could to clean up the dataset by removing labeling errors, which meant filtering out a lot of stuff from VGG.  I did this by repeatedly training a face recognition CNN and then using graph clustering methods and a lot of manual review to clean up the dataset.  In the end about half the images are from VGG and face scrub.  Also, the total number of individual identities in the dataset is 7485.  I made sure to avoid overlap with identities in LFW.

  The network training started with randomly initialized weights and used a structured metric loss that tries to project all the identities into non-overlapping balls of radius 0.6.  The loss is basically a type of pair-wise hinge loss that runs over all pairs in a mini-batch and includes hard-negative mining at the mini-batch level.

  The resulting model obtains a mean error of 0.993833 with a standard deviation of 0.00272732 on the LFW benchmark. 
  

* mmod_dog_hipsterizer.dat.bz2

  This dataset is trained on the data from the Columbia Dogs dataset, which was introduced in the paper:
  
      Dog Breed Classification Using Part Localization
      Jiongxin Liu, Angjoo Kanazawa, Peter Belhumeur, David W. Jacobs 
      European Conference on Computer Vision (ECCV), Oct. 2012. 
      
   The original dataset is not fully annotated.  So I created a new fully annotated version which is available here:  http://dlib.net/files/data/CU_dogs_fully_labeled.tar.gz

* mmod_human_face_detector.dat.bz2

  This is trained on this dataset: http://dlib.net/files/data/dlib_face_detection_dataset-2016-09-30.tar.gz.  
  I created the dataset by finding face images in many publicly available
  image datasets (excluding the FDDB dataset).  In particular, there are images
  from ImageNet, AFLW, Pascal VOC, the VGG dataset, WIDER, and face scrub.  
  
  All the annotations in the dataset were created by me using dlib's imglab tool.

* resnet34_1000_imagenet_classifier.dnn.bz2

  This is trained on the venerable ImageNet dataset.  
  
* shape_predictor_5_face_landmarks.dat.bz2
  
  This is a 5 point landmarking model which identifies the corners of the eyes and bottom of the nose.  It is 
  trained on the [dlib 5-point face landmark dataset](http://dlib.net/files/data/dlib_faces_5points.tar), which consists of
  7198 faces.  I created this dataset by downloading images from the internet and annotating them with dlib's imglab tool.
  
  The exact program that produced the model file can be found [here](https://github.com/davisking/dlib/blob/master/tools/archive/train_face_5point_model.cpp).
  
  This model is designed to work well with dlib's HOG face detector and the CNN face detector (the one in mmod_human_face_detector.dat). 
  
* shape_predictor_68_face_landmarks.dat.bz2
 
  This is trained on the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
  
      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
      300 faces In-the-wild challenge: Database and results. 
      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
   
   The license for this dataset excludes commercial use and Stefanos Zafeiriou,
   one of the creators of the dataset, asked me to include a note here saying
   that the trained model therefore can't be used in a commerical product.  So
   you should contact a lawyer or talk to Imperial College London to find out
   if it's OK for you to use this model in a commercial product.  
 
   Also note that this model file is designed for use with dlib's HOG face detector.  That is, it expects the bounding
   boxes from the face detector to be aligned a certain way, the way dlib's HOG face detector does it.  It won't work
   as well when used with a face detector that produces differently aligned boxes, such as the CNN based mmod_human_face_detector.dat face detector. 

* mmod_rear_end_vehicle_detector.dat.bz2
 
  This model is trained on the [dlib rear end vehicles dataset](http://dlib.net/files/data/dlib_rear_end_vehicles_v1.tar).  The dataset contains images from vehicle dashcams which I manually annotated using dlib's imglab tool.
  
* mmod_front_and_rear_end_vehicle_detector.dat.bz2

  This model is trained on the [dlib front and rear end vehicles dataset](http://dlib.net/files/data/dlib_front_and_rear_vehicles_v1.tar).  The dataset contains images from vehicle dashcams which I manually annotated using dlib's imglab tool.
  
