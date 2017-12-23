//
// Created by lex on 28.11.17.
//

#ifndef TEST_NNETWORK_H
#define TEST_NNETWORK_H
#include "typedefs.h"
#include <random>
//#include <gtest/gtest_prod.h>

class NNetwork {
public:
    NNetwork(int in_count, int hidden_count, int out_count);
    NNetwork(std::string filename);

    /*
     * Setup network parameters for training
     * Data is considered to be a 2D array of floats representing the image set
     * (image = 1d array, e.g. OpenCVed pic stored in Mat, parsed to plain array/stdvec)
     */
    void setup(vec2d &_data, vec2d& _validata, int _epoch_num, float _lrn_rate, float _epsilon, vec2d& _validation_set, ivec1d& _validation_labels);

    void train();

    void predict(vec1d &x);

    void save(std::string filename);

    int get_class();

private:
    //FRIEND_TEST(NNetworkTest, Forward);

    vec2d data;
    vec2d validata;
    vec2d validation_set;
    ivec1d validation_labels;

    int epoch_num;
    float learn_rate, epsilon;

    /*
     * Diagram explaining network layout:
     *
     * Input layer          Hidden Layer            Output Layer
     *          To Hidden               From Hidden
     *
     *      I ----------------- H ------------------ O
     *      I ----------------- H ------------------ O
     *      I ----------------- H ------------------ O
     *      I ----------------- H ------------------ O
     *
     */
    int input_size, hidden_size, output_size;

    vec1d input;
    vec1d hidden;
    vec1d output;

    vec2d tohidden_ws; // Weights between I-H layers
    vec2d fromhidden_ws; // Weights between H-O layers
    vec1d tohidden_bs; // Biases, same naming convention as weights
    vec1d fromhidden_bs;

    vec1d ho_gradients; // H-O layers gradients (or whatever those things are)

    /* Misc */
    std::mt19937 gen;//mersenne twister generator
    std::uniform_real_distribution<> dis;
    float rand0to1();

    inline float sigma(float x) { return static_cast<float>(1.0 / (1.0 + exp(-x))); }
    void softmax();
    void shuffle();


    void backpropagate(vec1d &y);

    float cross_entropy(vec1d &vector);

    void display_progress(int i, unsigned long size);
};


#endif //TEST_NNETWORK_H
