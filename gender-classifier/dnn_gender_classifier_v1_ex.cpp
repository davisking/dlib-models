// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
	This example shows "Minimalistic CNN-based model" for gender prediction from
	face images. The original idea and definition of the model is given in the
	document available at this address:
	http://www.eurecom.fr/fr/publication/4768/download/mm-publi-4768.pdf 
    
	This model is a gender classifier trained using a private dataset of about
	200k different face images, , with a balanced distribution between female
	and male genders. It was generated according to the network definition and
	settings given in the paper for gender prediction from face images. Even if
	the dataset used for the training is different from that used by G. Antipov
	et al, the classification results on the LFW evaluation are similar overall.
	
	To take up the authors' proposal to join the results of three networks, a
	simplification was made by finally presenting RGB images, thus simulating
	three "grayscale" networks via the three image planes. Better results could
	be probably obtained with a more complex and deeper network, but the performance
	of the classification is nevertheless surprising compared to the simplicity
	of the network used and thus its very small size.
	
	The model was trained using about 200k images, with a balanced distribution
	between female and male genders. Faces extracted from a non-public database
	were augmented, after alignment according to the 5-point model provided
	by Dlib. The CNN pretrained model he model was saved just after training so
	that if necessary it could be fine-tuned using another database.

	Finally, this gender model is provided for free by Cydral and is licensed
	under the Creative Commons Zero v1.0 Universal.
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

const char* VERSION = "1.1";

// ----------------------------------------------------------------------------------------

// This block of statements defines the network as proposed for the CNN Model I.
// We however removed the "dropout" regularization on the activations of convolutional layers.
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, stride, stride, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using res_ = relu<block<N, bn_con, 1, SUBNET>>;
template <int N, typename SUBNET> using ares_ = relu<block<N, affine, 1, SUBNET>>;

template <typename SUBNET> using alevel1 = avg_pool<2, 2, 2, 2, ares_<64, SUBNET>>;
template <typename SUBNET> using alevel2 = avg_pool<2, 2, 2, 2, ares_<32, SUBNET>>;

using agender_type = loss_multiclass_log<fc<2, multiply<relu<fc<16, multiply<alevel1<alevel2< input_rgb_image_sized<32>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 40
void display_progressbar(float percentage)
{
	uint32_t val = (int)(percentage * 100);
	uint32_t lpad = (int)(percentage * PBWIDTH);
	uint32_t rpad = PBWIDTH - lpad;
	printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
	fflush(stdout);
}

// ----------------------------------------------------------------------------------------

// An helper function to load dataset for testing.
enum label_ : uint16_t
{
	female_label,
	male_label,
};
struct image_info
{
	enum label_ label_image;
	string filename;
};

std::vector<image_info> get_image_listing(
	const std::string& images_folder,
	const enum label_& label
)
{
	std::vector<image_info> results;
	image_info temp;
	temp.label_image = label;

	auto dir = directory(images_folder);
	for (auto image_file : dir.get_files())
	{
		temp.filename = image_file;
		results.push_back(temp);
	}

	return results;
}
std::vector<image_info> get_train_listing_females(const std::string& ifolder) { return get_image_listing(ifolder, female_label); }
std::vector<image_info> get_train_listing_males(const std::string& ifolder) { return get_image_listing(ifolder, male_label); }

int main(int argc, char** argv) try {
	// Use a parser to set parameters.
	command_line_parser parser;

	parser.add_option("h", "");
	parser.add_option("help", "Displays this information");
	parser.add_option("version", "Display version");
	parser.add_option("test-folders", "Test the gender classifier ./females ./males", 2);
	parser.add_option("test-image", "Test the gender classifier ./image.jpg", 1);
	parser.parse(argc, argv);

	parser.check_incompatible_options("help", "version");
	parser.check_incompatible_options("help", "test-folders");
	parser.check_incompatible_options("help", "test-image");
	parser.check_incompatible_options("version", "test-folders");
	parser.check_incompatible_options("version", "test-image");
	parser.check_incompatible_options("test-folders", "test-image");

	if (parser.option("help") || parser.option("h"))
	{
		cout << "Usage: ./dnn_gender_classifier_ex [options]\n";
		parser.print_options(cout);
		cout << endl << "Note:" << endl << "You will need two different models to test this classifier.";
		cout << endl << "First, download and then decompress the gender model from: " << endl;
		cout << "http://dlib.net/files/dnn_gender_classifier_v1.dat.bz2" << endl;
		cout << "Then, you also need the face landmarking model file from:" << endl;
		cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
		cout << endl << endl;
		return EXIT_SUCCESS;
	}
	if (parser.option("version"))
	{
		cout << "dnn_gender_classifier_ex v" << VERSION
			 << "\nCompiled: " << __TIME__ << " " << __DATE__ << endl << endl;
		return EXIT_SUCCESS;
	}

	if (parser.option("test-folders"))
	{		
		// Initialize networks.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

		static const char* model_net_filename = "dnn_gender_classifier_v1.dat";
		agender_type net;
		deserialize(model_net_filename) >> net;

		// Load test images.
		auto female_images = get_train_listing_females(parser.option("test-folders").argument(0));
		cout << "female examples: " << female_images.size() << endl;

		auto male_images = get_train_listing_males(parser.option("test-folders").argument(1));
		cout << "male examples: " << male_images.size() << endl << endl;

		// Test all images.
		matrix<rgb_pixel> in;
		int32_t num_right = 0, num_wrong = 0, nb_elts = 0;		
		cout << "analysing female faces in progress: " << endl;
		for (auto& img : female_images)
		{
			display_progressbar((nb_elts++ / (float)female_images.size()));
			load_image(in, img.filename);
			for (auto face : detector(in))
			{
				auto shape = sp(in, face);
				if (shape.num_parts())
				{
					matrix<rgb_pixel> face_chip;
					extract_image_chip(in, get_face_chip_details(shape, 32), face_chip);
					if (net(face_chip) == female_label)
						++num_right;
					else
						++num_wrong;					
				}
			}			
		}
		display_progressbar(1.0f);
		// ---
		cout << endl << "analysing male faces in progress: " << endl;
		nb_elts = 0;
		for (auto& img : male_images)
		{
			display_progressbar((nb_elts++ / (float)male_images.size()));
			load_image(in, img.filename);
			for (auto face : detector(in))
			{
				auto shape = sp(in, face);
				if (shape.num_parts())
				{
					matrix<rgb_pixel> face_chip;					
					extract_image_chip(in, get_face_chip_details(shape, 32), face_chip);
					if (net(face_chip) == male_label)
						++num_right;
					else
						++num_wrong;
				}
			}			
		}
		display_progressbar(1.0f);
		cout << endl;

		// And at final, display accuracy.
		cout << "testing num_right: " << num_right << endl;
		cout << "testing num_wrong: " << num_wrong << endl;
		cout << "testing accuracy:  " << num_right / (double)(num_right + num_wrong) << endl;
		return EXIT_SUCCESS;
	}

	if (parser.option("test-image"))
	{
		// Initialize networks.
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor sp;
		deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

		static const char* model_net_filename = "dnn_gender_classifier_v1.dat";
		agender_type net;
		deserialize(model_net_filename) >> net;

		// Load the source image.
		matrix<rgb_pixel> in;
		load_image(in, parser.option("test-image").argument(0));

		// As proposed in the paper, use Softmax for the last layer.
		softmax<agender_type::subnet_type> snet;
		snet.subnet() = net.subnet();		

		// Evaluate the gender
		int32_t cur_face = 0;
		float confidence = 0.0f;
		enum label_ gender;
		for (auto face : detector(in))
		{
			auto shape = sp(in, face);
			if (shape.num_parts())
			{
				matrix<rgb_pixel> face_chip;
				extract_image_chip(in, get_face_chip_details(shape, 32), face_chip);
				matrix<float, 1, 2> p = mat(snet(face_chip));
				if (p(0) < p(1))
				{
					gender = male_label;
					confidence = p(1);
				}
				else
				{
					gender = female_label;
					confidence = p(0);
				}
				cout << "face#" << cur_face++ << " - detected gender: " << ((gender == female_label) ? "female" : "male");
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
