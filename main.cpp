#include <iostream>
#include <cmath>
#include <cstring>
#include "typedefs.h"
#include "NNetwork.h"
#include "Utils.h"

int main(int argc, char** argv) {

    if (argc > 1 && (strcmp(argv[1], "-h") == 0 ||strcmp(argv[1], "--help") == 0)){
        std::cout << "Usage: <exec_name> H L Err Epch TD TL VD VL S \nReplace each var with corresponding argument, where\n";
        std::cout << "\tH - number of neurons in hidden layer\n";
        std::cout << "\tL - learning rate for gradient descend opt method\n";
        std::cout << "\tErr - desired cross-entropy accuracy\n";
        std::cout << "\tEpch - number of epochs\n";
        std::cout << "\tTD - path to file with train data\n";
        std::cout << "\tTL - path to file with train labels\n";
        std::cout << "\tVD - path to file with validation data\n";
        std::cout << "\tVL - path to file with validation labels\n";
        std::cout << "\tS - path to file in which network params will be stored\n";
        std::cout << "For H L Err and Epch zero value can be used. In this case default values are as following:\n";
        std::cout << "\t200 neurons in hidden layer\n";
        std::cout << "\t0.01 as learning rate for gradient descend opt method\n";
        std::cout << "\t0.005 as desired cross-entropy accuracy\n";
        std::cout << "\t10 train epochs\n";
        std::cout << "If S is not specified, default name \"vanilla.params\" will be used" << std::endl;
        return 0;
    }

    int hidden_count = atoi(argv[1]);
    float learning_rate = static_cast<float>(atof(argv[2]));
    float ce_error = static_cast<float>(atof(argv[3]));
    int epochs = atoi(argv[4]);

    hidden_count = hidden_count == 0 ? 200 : hidden_count;
    learning_rate = learning_rate == 0.0f ? 0.01f : learning_rate;
    ce_error = ce_error == 0.0f ? 0.005f : ce_error;
    epochs = epochs == 0 ? 10 : epochs;

    std::string train_set(argv[5]);
    std::string train_labels(argv[6]);
    std::string test_set(argv[7]);
    std::string test_labels(argv[8]);
    std::string save_filename = "vanilla.params";
    if (argc > 9){
        save_filename = argv[9];
    }

    NNetwork network(28*28, hidden_count, 10);
    {
        vec2d data = Utils::read_Mnist(train_set);
        ivec1d labels = Utils::read_Mnist_Label(train_labels); //e.g. 1 8 0 4 8 7 5 3 etc
        vec2d validata = Utils::convert_labels(labels); //labels transformed into NN output, e.g. {0,1,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,1,0,0}
        network.setup(data, validata, epochs, learning_rate, ce_error);
    }
    network.train();
    network.save(save_filename);
    std::cout << "Network is trained. Saving net parameters to file..." << std::endl;
    std::cout << "Validation" << std::endl;

    vec2d validation_set = Utils::read_Mnist(test_set);
    ivec1d validation_labels = Utils::read_Mnist_Label(test_labels);
    float accuracy = Utils::validate(network, validation_set, validation_labels);
    std::cout << "Accurac on validation set is " << accuracy << std::endl;

    return 0;
}