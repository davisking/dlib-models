#include "resnet.h"

// In this simple example we will show how to load a pretrained network
// and use it for a different task.  In particular, we will load a ResNet50
// trained on ImageNet and use it as a pretrained backbone for some metric
// learning task

using namespace std;
using namespace dlib;

// here's an example of how one could define a metric learning network
// using the ResNet50's backbone from resnet.h
namespace model
{
    template<template<typename> class BN>
    using net_type = loss_metric<
        fc_no_bias<128,
        avg_pool_everything<
        typename resnet<BN>::template backbone_50<
        input_rgb_image
        >>>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}


int main() try
{

    // ResNet50 classifier trained on ImageNet
    resnet<bn_con>::l50 resnet50;
    std::vector<string> labels;
    deserialize("resnet/resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;

    // We can now assign ResNet50's backbone to our network skipping the different layers,
    // in our case, the loss layer and the fc layer:
    model::train net;
    net.subnet().subnet() = resnet50.subnet().subnet();

    // An alternative way to use the pretrained network on a different network is
    // to extract the relevant part of the network (we remove loss and fc layers),
    // stack the new layers on top of it and assign the network
    // To change the loss layer and the fc layer while keeping the pretrained backbone:
    auto backbone = resnet50.subnet().subnet();
    using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
    net_type net2;
    // copy the backbone to the newly defined network
    net2.subnet().subnet() = backbone;

    // From this point on, we can train the new network using this pretrained backbone.

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
