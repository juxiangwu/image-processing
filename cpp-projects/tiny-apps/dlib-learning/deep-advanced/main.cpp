// dlib dnn practice

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace dlib; // conv layer

// define a resnet block layer
template<int num_filter,
    template<typename> class BN,
    int stride,
    typename SUBNET>
    using block = BN < con<num_filter, 3, 3, 1, 1,
    relu<BN<con<num_filter, 3, 3, stride, stride,
    SUBNET>>>>>;

// define a resnet residual block
template<
    template<int, template<typename> class, int, typename> class block,
    int num_filter,
    template<typename> class BN,
    typename SUBNET>
    using residual = add_prev1<block<num_filter, BN, 1, tag1<SUBNET>>>;

// define downsampling (stride2) residual block
template<
    template<int, template<typename> class, int, typename> class block,
    int num_filter,
    template<typename> class BN,
    typename SUBNET>
    using residual_down = add_prev2 < avg_pool<2, 2, 2, 2,
    skip1<tag2<block<num_filter, BN, 2, tag1<SUBNET>>>>>>;

// now define 4 different residual blocks
template <typename SUBNET> using res = relu<residual<block, 8, bn_con, SUBNET>>;
template <typename SUBNET> using res_a = relu < residual<block, 8, affine, SUBNET>>;
template <typename  SUBNET> using res_down = relu<residual_down<block, 8, bn_con, SUBNET>>;
template <typename SUBNET> using resa_down = relu<residual_down<block, 8, affine, SUBNET>>;

// building the net type
const unsigned long num_classes = 10;
using net_type = loss_multiclass_log < fc<num_classes,
    avg_pool_everything<res<res<res<res_down<
    repeat<9, res,
    res_down<res<
    input<matrix<unsigned char>>>>>>>>>>>>;


int main(int argc, char**argv)
{
    // load data
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long> training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long> testing_labels;
    load_mnist_dataset("D:/Develop/DL/projects/resources/datasets/mnist", training_images, training_labels, testing_images, testing_labels);

    // use smaller ram
    set_dnn_prefer_smallest_algorithms();

    // create a net
    net_type net;

    // print layer info
    std::cout << "net has " << net.num_layers << " layers\n";
    //std::cout << net << "\n";

    // get output for layer 3 ==>  layer<3>(net).get_output()

    // now set trainer
    dnn_trainer<net_type, adam> trainer(net, adam(0.0005, 0.9, 0.999));
    trainer.be_verbose();
    trainer.set_iterations_without_progress_threshold(2000);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_learning_rate(0.001);
    trainer.set_synchronization_file("D:/Develop/DL/projects/digital-image-processing/temp/mnist_res_sync",
        std::chrono::seconds(100));

    // set mini batch
    std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels;
    dlib::rand rnd(time(0));

    while (trainer.get_learning_rate() >= 1e-6)
    {
        mini_batch_samples.clear();
        mini_batch_labels.clear();
        // make 128 mini batch
        while (mini_batch_samples.size() < 128)
        {
            auto idx = rnd.get_random_32bit_number() % training_images.size();
            mini_batch_samples.push_back(training_images[idx]);
            mini_batch_labels.push_back(training_labels[idx]);
        }
        // train mini batch
        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
        // can also use test_one_step to show test accuracy
    }

    // train_one_step is multithreaded implementation. So need to use
    // trainer.get_net to perform synchronization
    trainer.get_net();

    // save net
    net.clean();
    serialize("D:/Develop/DL/projects/digital-image-processing/temp/mnist_res_network.dat") << net;

    // test net: batchnorm will be replaced by affine layer
    using test_net_type = loss_multiclass_log < fc < num_classes,
        avg_pool_everything<res_a<res_a<res_a<resa_down<
        repeat<9, res_a,
        resa_down<res_a<
        input<matrix<unsigned char>>>>>>>>>>>>;

    // can assign trained net to test net
    test_net_type tnet = net;
    // or deserialize from saved file

    // run training data
    std::vector<unsigned long> predicted_labels = tnet(training_images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < training_images.size(); i++)
    {
        if (predicted_labels[i] == training_labels[i])
        {
            num_right++;
        }
        else
        {
            num_wrong++;
        }
    }
    std::cout << "training num right= " << num_right << "\n";
    std::cout << "training num wrong= " << num_wrong << "\n";
    std::cout << "training accuracy= " << num_right / double(num_right + num_wrong) << "\n";

    // run test data
    predicted_labels = tnet(testing_images);
    num_right = 0;
    num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        if (predicted_labels[i] == testing_labels[i])
        {
            num_right++;
        }
        else
        {
            num_wrong++;
        }
    }
    std::cout << "testing num right= " << num_right << "\n";
    std::cout << "testing num wrong= " << num_wrong << "\n";
    std::cout << "testing accuracy= " << num_right / double(num_right + num_wrong) << "\n";


    std::system("pause");
    return 0;
}
