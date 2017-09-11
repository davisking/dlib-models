# dlib-models
This repository contains trained models created by me (Davis King).  They are provided as part of the dlib example programs, which are intended to be educational documents that explain how to use various parts of the dlib library.  As far as I am concerned, anyone can do whatever they want with these model files.  However, the models are trained on datasets might fall under more restrictive licenses.  The details are summarized below for each model file.

* dlib_face_recognition_resnet_model_v1.dat.bz2
  
  This file is trained on about 3 million images I scraped from the internet.  The list of URLs was obtained via
  google searches as well as the lists of URLs provided by the VGG face dataset (http://www.robots.ox.ac.uk/~vgg/data/vgg_face/) and face scrub dataset (http://vintage.winklerbros.net/facescrub.html). 
  
  I spent a lot of time fixing the annotations and creating new annotations for new identities. 

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
  
* shape_predictor_68_face_landmarks.dat.bz2
 
  This is trained on the ibug 300-W dataset (https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
  
      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
      300 faces In-the-wild challenge: Database and results. 
      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
   
   The license for this dataset excludes commercial use.  So you should contact Imperial College London to find out if it's OK for you use use this model in a commercial product.
    
