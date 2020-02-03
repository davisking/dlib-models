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

class visitor_lr_multiplier
{
public:

    visitor_lr_multiplier(double new_lr_multiplier_) : new_lr_multiplier(new_lr_multiplier_) {}

    template <typename T>
    void set_learning_rate_multipler(T&) const
    {
        // ignore other layer detail types
    }

    template <layer_mode mode>
    void set_learning_rate_multipler(bn_<mode>& l) const
    {
        l.set_learning_rate_multiplier(new_lr_multiplier);
        l.set_bias_learning_rate_multiplier(new_lr_multiplier);
    }

    template <long nf, long nr, long nc, int sx, int sy>
    void set_learning_rate_multipler(con_<nf,nr,nc,sx,sy>& l) const
    {
        l.set_learning_rate_multiplier(new_lr_multiplier);
        l.set_bias_learning_rate_multiplier(new_lr_multiplier);
    }

    template<typename input_layer_type>
    void operator()(size_t , input_layer_type& )  const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t , add_layer<T,U,E>& l)  const
    {
        set_learning_rate_multipler(l.layer_details());
    }

private:

    double new_lr_multiplier;
};


int main() try
{

    // ResNet50 classifier trained on ImageNet
    resnet<bn_con>::l50 resnet50;
    std::vector<string> labels;
    deserialize("resnet/resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;

    // The ResNet50 backbone
    auto backbone = resnet50.subnet().subnet();

    // We can now assign ResNet50's backbone to our network skipping the different layers,
    // in our case, the loss layer and the fc layer:
    model::train net;
    net.subnet().subnet() = backbone;

    // An alternative way to use the pretrained network on a different network is
    // to extract the relevant part of the network (we remove loss and fc layers),
    // stack the new layers on top of it and assign the network
    using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
    net_type net2;

    // copy the backbone to the newly defined network
    net2.subnet().subnet() = backbone;

    // now we are going to adjust the learning rates of different layers
    visit_layers_range<  2,  37, net_type, visitor_lr_multiplier>(net2, visitor_lr_multiplier(0.1));
    visit_layers_range< 38, 106, net_type, visitor_lr_multiplier>(net2, visitor_lr_multiplier(0.01));
    visit_layers_range<107, 153, net_type, visitor_lr_multiplier>(net2, visitor_lr_multiplier(0.001));
    visit_layers_range<154, 192, net_type, visitor_lr_multiplier>(net2, visitor_lr_multiplier(0.0001));

    // check the results
    cout << net2 << endl;

    // From this point on, we can finetune the new network using this pretrained backbone.

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
