#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace dlib;

/* ------ Define the CNN for face detection -------------
 target input image is 50 x 50
 3 downsampling layers (8x reduction)
 4 conv layers
*/

// 5x5 conv with 2x downsampling block
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
// 3x3 conv no downsampling block
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;
// 8x downsampling block using 3 5x5 conv, 32 channels
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32,
    relu<bn_con<con5d<32,
    relu<bn_con<con5d<32,SUBNET>>>>>>>>>;
// rest of the network: 3x3 conv with batch normalization
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

// finally add one-channel, 6x6 classifier layer and loss_mmod to complete the network
using net_type = loss_mmod<con<1, 6, 6, 1, 1,
    rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// --- main ----
int main(int argc, char** argv)
{
    const std::string face_dir = "../data/faces";

    // load dataset
    std::vector<matrix<rgb_pixel>> train_images, test_images;
    std::vector<std::vector<mmod_rect>> train_boxes, test_boxes;

    load_image_dataset(train_images, train_boxes, face_dir+"/training.xml");
    load_image_dataset(test_images, test_boxes, face_dir+"/testing.xml");

    std::cout << "num train images= " << train_images.size() << "\n";
    std::cout << "num of test imamges= " << test_images.size() << "\n";

    // set mmod options- pick the minimal size of face in pixels
    mmod_options options(train_boxes, 40, 40);
    std::cout << "num detector windows" << options.detector_windows.size() << "\n";
    for (auto w : options.detector_windows)
        std::cout << "detector window width by height: " << w.width << " x " << w.height << "\n";
    std::cout << "overlap NMS IOU thresh: " << options.overlaps_nms.get_iou_thresh() << "\n";
    std::cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << "\n";

    // create the net
    net_type net(options);
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    // setup the trainer
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("./MaxMarginObjectDetection.dir/mmod_sync",
        std::chrono::minutes(2));
    trainer.set_iterations_without_progress_threshold(50);

    // train the network
    std::vector<matrix<rgb_pixel>> mini_batch_images;
    std::vector<std::vector<mmod_rect>> mini_batch_labels;

    // create a random image cropper. the cropper will perform
    // rotation, cropping, and translation
    random_cropper cropper;
    cropper.set_chip_dims(200,200);
    cropper.set_min_object_size(0.2,0.2);
    dlib::rand rnd;

    // training loop
    int num_minibatch= 30;
    while (trainer.get_learning_rate() >= 1e-4)
    {
        cropper(num_minibatch, train_images, train_boxes,
            mini_batch_images, mini_batch_labels);
        // also apply random color jittering
        for (auto &img: mini_batch_images)
            disturb_colors(img, rnd);

        // train one step
        trainer.train_one_step(mini_batch_images, mini_batch_labels);
    }

    // wait for the training threads...
    trainer.get_net();
    std::cout<<"Done training!\n";

    // save network to disk
    net.clean();
    serialize("./MaxMarginObjectDetection.dir/mmod_network.dat") << net;

    // test on training data
    std::cout<<"training results: "<<
        test_object_detection_function(net, train_images, train_boxes)<<"\n";
    // test on testing data
    std::cout<<"test results: "<<
        test_object_detection_function(net, test_images, test_boxes)<<"\n";

    // print the trainer and cropper settings
    std::cout<<trainer<< cropper<<"\n";

    // look at the results on test images
    image_window win;
    for (auto&& img: test_images)
    {
        pyramid_up(img);
        auto dets= net(img);  // get detections using forward prop
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d: dets)
        {
            std::cout << "detection confidence= " << d.detection_confidence
                << " rect= " << d.rect.top();
            win.add_overlay(d);
        }
        std::cin.get();  // wait for user to skip to the next image
    }

    std::system("pause");
    return 0;
}
