//
// Created by lex on 12.11.17.
//
#include "../typedefs.h"
#include "../NNetwork.h"
#include "../Utils.h"
#include <gtest/gtest.h>


TEST(NNetworkTest, Shuffle) {
    vec2d data = { {1}, {2}, {3}, {4} };
    vec2d validata = { {1}, {2}, {3}, {4} };

    auto n = static_cast<int>(data.size());
    for (int i=0; i<n; i++){
        int idx1 = rand() % n;
        int idx2 = rand() % n;
        std::swap(data[idx1], data[idx2]);
        std::swap(validata[idx1], validata[idx2]);
    }

    for (int i=0; i<4; i++){
        ASSERT_EQ(data[i][0], validata[i][0]);
    }
}

TEST(NNetworkTest, GetClass){
    vec1d output = { 0.02f, 0.34f, 0.12f, 0.02f, 0.5f };
    auto _max = std::max_element(output.begin(), output.end());
    int i = static_cast<int>(std::distance(output.begin(), _max));
    ASSERT_EQ(4, i);
}

TEST(NNetworkTest, ConverLabels){
    ivec1d labels = { 0, 7, 8, 2};
    vec2d validata = Utils::convert_labels(labels);
    ASSERT_FLOAT_EQ(validata[0][0], 1.0f);
    ASSERT_FLOAT_EQ(validata[1][7], 1.0f);
    ASSERT_FLOAT_EQ(validata[2][8], 1.0f);
    ASSERT_FLOAT_EQ(validata[3][2], 1.0f);
}

TEST(NNetworkTest, Softmax){
    vec1d output = { 0, 0.6f, 1, -1.2f, 4};
    float max = *std::max_element(output.begin(), output.end());

    float div = 0.0f;
    for (int i = 0; i < output.size(); i++) {
        div += exp(output[i] - max);
    }

    for (int i = 0; i < output.size(); i++) {
        output[i] = exp(output[i] - max) / div;
    }
    float sum=0;
    for (auto f : output) sum +=f;
    std::cout << "Softmax for { 0, 0.6f, 1, -1.2f, 4} " << output[0] << " " << output[1] << " "  << output[2] << " "  << output[3] << " "  << output[4] << std::endl;
    ASSERT_FLOAT_EQ(1.0f, sum);
}

TEST(NNetworkTest, Forward){
    NNetwork network(2,2,2);

    network.tohidden_ws[0][0] = 0.5f;
    network.tohidden_ws[0][1] = 0.5f;
    network.tohidden_ws[1][0] = 0.5f;
    network.tohidden_ws[1][1] = 0.5f;
    float s = 1.0f/(1+exp(-4));
    network.fromhidden_ws[0][0] = 1.0f/s;
    network.fromhidden_ws[0][1] = 4.0f/s;
    network.fromhidden_ws[1][0] = 3.0f/s;
    network.fromhidden_ws[1][1] = 2.0f/s;
    network.tohidden_bs[0] = 1;
    network.tohidden_bs[1] = 1;
    network.fromhidden_bs[0] = 2;
    network.fromhidden_bs[1] = 2;

    vec1d x = {2,4};
    network.predict(x);

    vec1d output = {6, 8};

    float div = 0.0f;
    for (int i = 0; i < output.size(); i++) {
        div += exp(output[i]);
    }

    for (int i = 0; i < output.size(); i++) {
        output[i] = exp(output[i]) / div;
    }

    ASSERT_FLOAT_EQ(output[0], network.output[0]);
    ASSERT_FLOAT_EQ(output[1], network.output[1]);
}

TEST(NNetwork, PredictionTest){
    //considering the output equal to the input
    srand(0);
    int size = 1000;
    int valsize = 100;
    int epochs = 10;
    float lrn_rate = 0.05;
    NNetwork network(4, 25, 4);
    vec2d data;
    for (int i=0; i<size; i++){
        int r = rand()%4;
        vec1d t = { 0,0,0,0};
        t[r] = 1;
        data.emplace_back(t);
    }
    vec2d validata = data;
    network.setup(data, validata, epochs, lrn_rate, 0.005);
    network.train();
    vec2d validation_set;
    ivec1d validation_labels;
    for (int i=0; i<valsize; i++){
        int r = rand()%4;
        vec1d t = { 0,0,0,0};
        t[r] = 1;
        validation_set.emplace_back(t);
        validation_labels.push_back(r);
    }
    float accuracy = Utils::validate(network, validation_set, validation_labels);
    std::cout << "GTEST: Accuracy = " << accuracy << std::endl;
    ASSERT_GT(accuracy, 0.75f);
}

//TEST(NNetwork, PredictionTest2){
//    NNetwork network("vanilla.params");
//    vec2d validation_set = Utils::read_Mnist("../data/test_images");
//    ivec1d validation_labels = Utils::read_Mnist_Label("../data/test_labels");
//    float accuracy = Utils::validate(network, validation_set, validation_labels);
//    ASSERT_GT(accuracy, 0.75f);
//}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}