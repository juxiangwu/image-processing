#include <iostream>
#include <dlib/svm_threaded.h>

// define types
typedef dlib::matrix<double, 0, 1> col_vec;
// pair def: first is LHS, second is RHS
typedef std::pair<std::vector<col_vec>, std::vector<col_vec>> sample_type;
// associate info between LHS and RHS
typedef std::vector<long> label_type;

// all LHS and RHS are 3-dimensional vector in this example
const unsigned long num_dims = 3;

// pre define
void make_data(std::vector<sample_type> &samples,
               std::vector<label_type> &labels);

// feature extractor
struct feature_extractor
{
    typedef col_vec feature_vector_type;
    typedef col_vec lhs_element;
    typedef col_vec rhs_element;

    unsigned long num_features() const
    {
        return num_dims;
    }

    void get_features(const lhs_element &left,
                      const rhs_element &right,
                      feature_vector_type &feats) const
    {
        feats= dlib::squared(left - right);
    }
};

// serialization - empty since no state
void serialize(const feature_extractor&, std::ostream&) {}
void deserialize(const feature_extractor&, std::istream&) {}

int main()
{
    try
    {
        // get small bit of training data
        std::vector<sample_type> samples;
        std::vector<label_type> labels;
        make_data(samples, labels);

        // trainer
        dlib::structural_assignment_trainer<feature_extractor> trainer;
        // regularization constant c , large c==> overfit
        trainer.set_c(10);
        // use multiple cpu cores
        trainer.set_num_threads(4);

        // train now
        dlib::assignment_function<feature_extractor> assigner = trainer.train(samples, labels);

        // test the assigner on our data
        std::cout << "Test the learned assignment function" << std::endl;
        for (unsigned long i = 0; i < samples.size(); ++i)
        {
            std::vector<long> predicted_assignments = assigner(samples[i]);
            std::cout << "True labels: " << dlib::trans(dlib::mat(labels[i])) << std::endl;
            std::cout << "Predicted labels: " << dlib::trans(dlib::mat(predicted_assignments)) << std::endl;
        }

        // training accuracy
        std::cout << "training accuracy: " << dlib::test_assignment_function(assigner, samples, labels) << std::endl;


    }
    catch (std::exception &e)
    {
        std::cout << "Exception thrown" << std::endl;
        std::cout << e.what() << std::endl;
    }

    std::system("pause");
    return 0;
}

// make data function
void make_data(std::vector<sample_type> &samples,
               std::vector<label_type> &labels)
{
    // make four different vectors
    col_vec A(num_dims), B(num_dims), C(num_dims), D(num_dims);
    A = 1, 0, 0;
    B = 0, 1, 0;
    C = 0, 0, 1;
    D = 0, 1, 1;

    std::vector<col_vec> lhs;
    std::vector<col_vec> rhs;
    label_type mapping;

    lhs.resize(3);
    lhs[0] = A;
    lhs[1] = B;
    lhs[2] = C;

    rhs.resize(3);
    rhs[0] = B;
    rhs[1] = A;
    rhs[2] = C;

    mapping.resize(3);
    mapping[0] = 1;  // lhs 0 matches rhs 1
    mapping[1] = 0;
    mapping[2] = 2;

    samples.push_back(make_pair(lhs, rhs));
    labels.push_back(mapping);
    // ----------------
    lhs[0] = C;
    lhs[1] = A;
    lhs[2] = B;

    rhs[0] = A;
    rhs[1] = B;
    rhs[2] = D;

    mapping[0] = -1;
    mapping[1] = 0;
    mapping[2] = 1;

    samples.push_back(make_pair(lhs, rhs));
    labels.push_back(mapping);

    // -----------------
    lhs[0] = A;
    lhs[1] = B;
    lhs[2] = C;

    rhs.resize(4);
    rhs[0] = C;
    rhs[1] = B;
    rhs[2] = A;
    rhs[3] = D;

    mapping[0] = 2;
    mapping[1] = 1;
    mapping[2] = 0;

    samples.push_back(make_pair(lhs, rhs));
    labels.push_back(mapping);
    //-------------------------------
    lhs.resize(2);
    lhs[0] = B;
    lhs[1] = C;

    rhs.resize(3);
    rhs[0] = C;
    rhs[1] = A;
    rhs[2] = D;

    mapping.resize(2);
    mapping[0] = -1;
    mapping[1] = 0;

    samples.push_back(make_pair(lhs, rhs));
    labels.push_back(mapping);

    //-------------------------
    lhs.resize(3);
    lhs[0] = D;
    lhs[1] = B;
    lhs[2] = C;

    // rhs will be empty.  So none of the items in lhs can match anything.
    rhs.resize(0);

    mapping.resize(3);
    mapping[0] = -1;
    mapping[1] = -1;
    mapping[2] = -1;

    samples.push_back(make_pair(lhs, rhs));
    labels.push_back(mapping);
}
