// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
	The initial source for the model's creation came from the document of 
	Z. Qawaqneh et al.: "Deep Convolutional Neural Network for Age Estimation
	based on VGG-Face Model". However, our research has led us to significant
	improvements in the CNN model, allowing us to estimate the age of a person
	outperforming the state-of-the-art results in terms of the exact accuracy
	and for 1-off accuracy.
    
	This model is thus an age predictor leveraging a ResNet-10 architecture and
	trained using a private dataset of about 110k different labelled images.
	During the training, we used an optimization and data augmentation pipeline
	and considered several sizes for the entry image.

	Finally, this age predictor model is provided for free by Cydral and is
	licensed under the Creative Commons Zero v1.0 Universal.
*/

#include "dlib/data_io.h"
#include "dlib/string.h"
#include <dlib/cmd_line_parser.h>
#include <dlib/image_transforms.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include <iostream>
#include <iterator>

using namespace std;
using namespace dlib;

const char* VERSION = "1.0";

// ----------------------------------------------------------------------------------------

// This block of statements defines a Resnet-10 architecture for the age predictor.
// We will use 81 classes (0-80 years old) to predict the age of a face.
const unsigned long number_of_age_classes = 81;

// The resnet basic block.
template<
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	int stride,
	typename SUBNET
>
using basicblock = BN<con<num_filters, 3, 3, 1, 1, relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

// A residual making use of the skip layer mechanism.
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,  // some kind of batch normalization or affine layer
	typename SUBNET
> // adds the block to the result of tag1 (the subnet)
using residual = add_prev1<BLOCK<num_filters, BN, 1, tag1<SUBNET>>>;

// A residual that does subsampling (we need to subsample the output of the subnet, too).
template<
	template<int, template<typename> class, int, typename> class BLOCK,  // a basic block defined before
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<BLOCK<num_filters, BN, 2, tag1<SUBNET>>>>>>;

// Residual block with optional downsampling and batch normalization.
template<
	template<template<int, template<typename> class, int, typename> class, int, template<typename>class, typename> class RESIDUAL,
	template<int, template<typename> class, int, typename> class BLOCK,
	int num_filters,
	template<typename> class BN,
	typename SUBNET
>
using residual_block = relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

template<int num_filters, typename SUBNET>
using aresbasicblock_down = residual_block<residual_down, basicblock, num_filters, affine, SUBNET>;

// Some useful definitions to design the affine versions for inference.
template<typename SUBNET> using aresbasicblock256 = residual_block<residual, basicblock, 256, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock128 = residual_block<residual, basicblock, 128, affine, SUBNET>;
template<typename SUBNET> using aresbasicblock64  = residual_block<residual, basicblock, 64, affine, SUBNET>;

// Common input for standard resnets.
template<typename INPUT>
using aresnet_input = max_pool<3, 3, 2, 2, relu<affine<con<64, 7, 7, 2, 2, INPUT>>>>;

// Resnet-10 architecture for estimating.
template<typename SUBNET>
using aresnet10_level1 = aresbasicblock256<aresbasicblock_down<256, SUBNET>>;
template<typename SUBNET>
using aresnet10_level2 = aresbasicblock128<aresbasicblock_down<128, SUBNET>>;
template<typename SUBNET>
using aresnet10_level3 = aresbasicblock64<SUBNET>;
// The resnet 10 backbone.
template<typename INPUT>
using aresnet10_backbone = avg_pool_everything<
	aresnet10_level1<
	aresnet10_level2<
	aresnet10_level3<
	aresnet_input<INPUT>>>>>;

using apredictor_t = loss_multiclass_log<fc<number_of_age_classes, aresnet10_backbone<input_rgb_image>>>;

// ----------------------------------------------------------------------------------------

// Helper function to estimage the age
uint8_t get_estimated_age(matrix<float, 1, number_of_age_classes>& p, float& confidence)
{
	float estimated_age = (0.25f * p(0));
	confidence = p(0);

	for (uint16_t i = 1; i < number_of_age_classes; i++) {
		estimated_age += (static_cast<float>(i) * p(i));
		if (p(i) > confidence) confidence = p(i);
	}

	return std::lround(estimated_age);
}

int main(int argc, char** argv) try {
	// Use a parser to set parameters.
	command_line_parser parser;

	parser.add_option("h", "");
	parser.add_option("help", "Displays this information");
	parser.add_option("version", "Display version");
	parser.add_option("predict-age", "Predict the age of a person ./image.jpg", 1);
	parser.parse(argc, argv);

	parser.check_incompatible_options("help", "version");
	parser.check_incompatible_options("help", "predict-age");
	parser.check_incompatible_options("version", "predict-age");

	if (parser.option("help") || parser.option("h"))
	{
		cout << "Usage: ./dnn_age_predictor_v1_ex [options]\n";
		parser.print_options(cout);
		cout << endl << "Note:" << endl << "You will need two different models to predict the age.";
		cout << endl << "First, download and then decompress the age predictor model from: " << endl;
		cout << "http://dlib.net/files/dnn_age_predictor_v1.dat.bz2" << endl;
		cout << "Then, you also need the face landmarking model file from:" << endl;
		cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
		cout << endl << endl;
		return EXIT_SUCCESS;
	}
	if (parser.option("version"))
	{
		cout << "dnn_age_predictor_v1_ex v" << VERSION
			 << "\nCompiled: " << __TIME__ << " " << __DATE__ << endl << endl;
		return EXIT_SUCCESS;
	}

	if (parser.option("predict-age"))
	{
		// Initialize networks.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

		static const char* model_net_filename = "dnn_age_predictor_v1.dat";
		apredictor_t net;
		deserialize(model_net_filename) >> net;

		// Load the source image.
		matrix<rgb_pixel> in;
		load_image(in, parser.option("predict-age").argument(0));

		// Usea Softmax for the last layer to estimate the age.
		softmax<apredictor_t::subnet_type> snet;
		snet.subnet() = net.subnet();		

		// Age prediction using machine learning.
		int32_t cur_face = 0;		
		for (auto face : detector(in))
		{
			auto shape = sp(in, face);
			if (shape.num_parts())
			{
				float confidence;
				matrix<rgb_pixel> face_chip;				
				extract_image_chip(in, get_face_chip_details(shape, 64), face_chip);
				matrix<float, 1, number_of_age_classes> p = mat(snet(face_chip));
				cout << "face#" << cur_face++ << " - age prediction: " << to_string(get_estimated_age(p, confidence)) << " years old";
				cout << std::fixed << std::setprecision(1) << " [" << (confidence * 100.0f) << "%]" << endl;
			}
		}
		return EXIT_SUCCESS;
	}
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
