//
// Created by lex on 02.12.17.
//

#ifndef TEST_UTILS_H
#define TEST_UTILS_H
#include "typedefs.h"
#include <fstream>
#include <vector>

/**
 * This class handles I/O, training and validation of Neural Network.
 * I prefer to keep NN clear of any application-related stuff, e.g. decoding the output as a mnist number.
 * This class does it.
 * Reading MNIST code is stolen from here http://eric-yuan.me/cpp-read-mnist/ only slightly modified
 * I hope it's not a big deal since we study deep learning, not the binary I/O :)
 */
class Utils {

private:
    static int ReverseInt(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }



public:
    static vec2d convert_labels(ivec1d& vec){
        vec1d layout = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        vec2d labels;   labels.reserve(vec.size());
        for (auto i : vec){
            auto out = layout;
            out[i] = 1.0;
            labels.push_back(out);
        }
        return labels;
    }

    static vec2d read_Mnist(std::string filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open())
        {
            vec2d vec;
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            file.read((char*)&n_rows, sizeof(n_rows));
            n_rows = ReverseInt(n_rows);
            file.read((char*)&n_cols, sizeof(n_cols));
            n_cols = ReverseInt(n_cols);
            vec.reserve(static_cast<unsigned long>(number_of_images));
            for (int i = 0; i < number_of_images; ++i)
            {
                vec1d tp;
                for (int r = 0; r < n_rows; ++r)
                {
                    for (int c = 0; c < n_cols; ++c)
                    {
                        unsigned char temp;
                        file.read((char*)&temp, sizeof(temp));
                        tp.push_back((float)temp/255.0f);
                    }
                }
                vec.push_back(tp);
            }
            return vec;
        }
        throw std::invalid_argument("Error while reading data from file " + filename);
    }

    static ivec1d read_Mnist_Label(std::string filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open())
        {
            ivec1d vec;
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;
            file.read((char*)&magic_number, sizeof(magic_number));
            magic_number = ReverseInt(magic_number);
            file.read((char*)&number_of_images, sizeof(number_of_images));
            number_of_images = ReverseInt(number_of_images);
            vec.resize(static_cast<unsigned long>(number_of_images));
            for (int i = 0; i < number_of_images; ++i)
            {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                vec[i] = (int)temp;
            }
            return vec;
        }
        throw std::invalid_argument("Error while reading labels from file " + filename);
    }


    static float validate(NNetwork& network, vec2d& data, ivec1d& labels) {
        float accuracy = 0.0f;
        int good = 0;
        int bad = 0;
        auto _x = data.begin();
        auto _y = labels.begin();
        for (; _x != data.end() && _y != labels.end(); _x++, _y++){
            auto& x = *_x;
            auto& y = *_y;
            network.predict(x);
            int u = network.get_class();
            if (u == y){
                good++;
            }
            else{
                bad++;
            }
        }
        return (float)good/data.size();
    }

};


#endif //TEST_UTILS_H
