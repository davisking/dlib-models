#include "resnet.h"

using namespace std;
using namespace dlib;

// here's an example of how one could define a custom network
// using the ResNet50's backbone from resnet.h
namespace model
{
    template<template<typename> class BN>
    using net_type = loss_metric<
        fc_no_bias<128,
        avg_pool_everything<
        typename resnet<BN>::template backbone_50<
        input_rgb_image_sized<227>
        >>>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}


int main() try
{
    { // we can define a custom model using the definition above
        model::train net;  // training model
        model::infer tnet; // testing model for inference
    }

    // However in this example we will learn how to load the
    // ImageNet pretrained ResNet50 model from disk.
    resnet<bn_con>::l50 resnet50;
    std::vector<string> labels;
    deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;

    // To change the loss layer and the fc layer while keeping the pretrained backbone:
    auto backbone = resnet50.subnet().subnet();
    using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
    net_type net;
    // copy the backbone to the newly defined network
    net.subnet().subnet() = backbone;

    // From this point on, we can train the new network using this pretrained backbone.

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
