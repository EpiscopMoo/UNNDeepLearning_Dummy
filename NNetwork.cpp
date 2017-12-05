//
// Created by lex on 01.12.17.
//

#include <algorithm>
#include <iostream>
#include <fstream>
#include "NNetwork.h"

NNetwork::NNetwork(int in_count, int hidden_count, int out_count) :
        input_size(in_count),
        hidden_size(hidden_count),
        output_size(out_count),
        gen(std::random_device()())
{
    input.resize(input_size);
    hidden.resize(hidden_size);
    output.resize(output_size);

    tohidden_bs.resize(hidden_size);
    fromhidden_bs.resize(output_size);

    tohidden_ws.resize(input_size, vec1d(hidden_size));
    fromhidden_ws.resize(hidden_size, vec1d(output_size));

    ho_gradients.resize(output_size, 0.0f);

    // Random initializing with floats in [0, 1)
    for (int i=0; i<hidden_size; i++){
        tohidden_bs[i] = rand0to1();
        for (int j=0; j<input_size; j++){
            tohidden_ws[j][i] = rand0to1();
        }
        for (int j=0; j<output_size; j++){
            fromhidden_ws[i][j] = rand0to1();
        }
    }
    std::for_each(fromhidden_bs.begin(), fromhidden_bs.end(), [this](float& x){x=rand0to1();});
}

void NNetwork::setup(vec2d &_data, vec2d& _validata, int _epoch_num, float _lrn_rate, float _epsilon)
{
    data = _data;
    validata = _validata;
    epoch_num  =_epoch_num;
    learn_rate = _lrn_rate;
    epsilon = _epsilon;
}

void NNetwork::train() {

    for (int echpochmak = 0; echpochmak < epoch_num; echpochmak++)
    {
        std::cout << "Running EPOCH " << echpochmak << std::endl;
        shuffle();
        float error = 0.0f;
        for (int i = 0; i<data.size(); i++)
        {
            display_progress(i, data.size());
            vec1d& x = data[i];
            vec1d& y = validata[i];
            predict(x);
            error += cross_entropy(y);
            backpropagate(y);
        }
        error /= data.size();
        error = -error;
        if (error < epsilon){
            std::cout << "Cross-entropy desired accuracy reached: " << error << " Stopping." << std::endl;
            return;
        }
    }

}

void NNetwork::shuffle() {
    auto n = static_cast<int>(data.size());
    for (int i=0; i<n; i++){
        int idx1 = rand() % n;
        int idx2 = rand() % n;
        std::swap(data[idx1], data[idx2]);
        std::swap(validata[idx1], validata[idx2]);
    }
}

void NNetwork::predict(vec1d &x) {
    input = x;
    //transition I --> H
    for (int i=0; i<hidden_size; i++){ //per each in Hidden layer
        auto& h = hidden[i]; //i-th neuron in hidden layer
        h = tohidden_bs[i];
        for (int w_idx = 0; w_idx < input_size; w_idx++){ //each in Input layer
            h += tohidden_ws[w_idx][i]*input[w_idx];
        }
        h = sigma(h);
    }
    //transition H --> O
    for (int i=0; i<output_size; i++){
        auto& u = output[i]; //i-th neuron in outer layer
        u = fromhidden_bs[i];
        for (int w_idx = 0; w_idx < hidden_size; w_idx++){
            u += fromhidden_ws[w_idx][i]*hidden[w_idx];
        }
    }
    softmax();
}

void NNetwork::backpropagate(vec1d &y) {
    /* Used cross entropy deriv. with softmax is y - u,
     * reference I used: http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
     */

    // 1. Compute "errors" for outputs
    for (int i=0; i<output_size; i++){
        ho_gradients[i] = -output[i] + y[i];

        // ... and update the weights in HO-layer accordingly
        for (int j=0; j<hidden_size; j++){
            fromhidden_ws[j][i] += learn_rate*0.9f * ho_gradients[i] * hidden[j];
        }
        fromhidden_bs[i] += learn_rate*0.9f * ho_gradients[i];
    }

    // 2. Now the same for IH-layer
    for (int j=0; j<hidden_size; j++){
        float ih_gradient = 0.0f;
        for (int k=0; k<output_size; k++){
            ih_gradient += ho_gradients[k]*fromhidden_ws[j][k];
        }
        ih_gradient *= hidden[j]*(1-hidden[j]);

        //update weights
        for (int i=0; i<input_size; i++){
            tohidden_ws[i][j] += learn_rate * ih_gradient * input[i];
        }
        tohidden_bs[j] += learn_rate * ih_gradient;
    }
}

float NNetwork::cross_entropy(vec1d &y) {
    float error = 0.0f;
    for (int i = 0; i < output_size; i++) {
        error += log(output[i]) * y[i];
    }
    return error;
}

void NNetwork::display_progress(int i, unsigned long size) {
    int stride = 10; // show message each 10%
    int portion = size / stride;
    int rem = i % portion;
    if (rem == 0){
        int percentage = i / portion * stride;
        std::cout << percentage << "% Done" << std::endl;
    }
}

void NNetwork::save(std::string filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if (file.is_open()){
        //write dimensions of the network: I size, H size, O size
        file.write((char*)&input_size, sizeof(int));
        file.write((char*)&hidden_size, sizeof(int));
        file.write((char*)&output_size, sizeof(int));

        //write biases, IH and HO
        file.write((char*)&tohidden_bs[0], sizeof(float)*tohidden_bs.size());
        file.write((char*)&fromhidden_bs[0], sizeof(float)*fromhidden_bs.size());

        //write weights
        for (auto& v : tohidden_ws){
            file.write((char*)&v[0], sizeof(float)*v.size());
        }
        for (auto& v : fromhidden_ws){
            file.write((char*)&v[0], sizeof(float)*v.size());
        }
        file.close();
    }
}

NNetwork::NNetwork(std::string filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if(file.is_open()){
        file.read((char*)&input_size, sizeof(int));
        file.read((char*)&hidden_size, sizeof(int));
        file.read((char*)&output_size, sizeof(int));

        input.resize(input_size);
        hidden.resize(hidden_size);
        output.resize(output_size);

        tohidden_bs.resize(hidden_size);
        fromhidden_bs.resize(output_size);

        tohidden_ws.resize(input_size, vec1d(hidden_size));
        fromhidden_ws.resize(hidden_size, vec1d(output_size));

        ho_gradients.resize(output_size, 0.0f);

        //write biases, IH and HO
        file.read((char*)&tohidden_bs[0], sizeof(float)*tohidden_bs.size());
        file.read((char*)&fromhidden_bs[0], sizeof(float)*fromhidden_bs.size());

        //write weights
        for (auto& v : tohidden_ws){
            file.read((char*)&v[0], sizeof(float)*v.size());
        }
        for (auto& v : fromhidden_ws){
            file.read((char*)&v[0], sizeof(float)*v.size());
        }
        file.close();
    }
}

int NNetwork::get_class() {
    auto _max = std::max_element(output.begin(), output.end());
    return static_cast<int>(std::distance(output.begin(), _max));
}

void NNetwork::softmax() {
    float div = 0.0f;
    for (int i = 0; i < output_size; i++) {
        div += exp(output[i]);
    }

    for (int i = 0; i < output_size; i++) {
        output[i] = exp(output[i]) / div;
    }

}

float NNetwork::rand0to1() {
    float num = (float)dis(gen);
    float factor = (float)dis(gen);
    return factor < 0.5f ? -num : num;
}
