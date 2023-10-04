#include <ggml.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
/**
Layer: my_model/embedding/embeddings:0 with shape (66, 256)
Layer: my_model/gru/gru_cell/kernel:0 with shape (256, 3072)
Layer: my_model/gru/gru_cell/recurrent_kernel:0 with shape (1024, 3072)
Layer: my_model/gru/gru_cell/bias:0 with shape (2, 3072)
Layer: my_model/dense/kernel:0 with shape (1024, 66)
Layer: my_model/dense/bias:0 with shape (66,)
**/

void print_2d(ggml_tensor * ts){
    for (int i1 = 0; i1 < ts->ne[1]; i1++){
        for (int i0 = 0; i0 < ts->ne[0]; i0++){
            cout << ggml_get_f32_1d(ts, i1*ts->ne[0] + i0) << " ";    
        }
        cout << endl;
    }   
}


struct rnn_generator {
    struct ggml_context * ctx;
    struct ggml_tensor * embeddings;
    struct ggml_tensor * cell_kernel;
    struct ggml_tensor * cell_recurrent_kernel;
    struct ggml_tensor * cell_bias;
    struct ggml_tensor * dense_kernel;
    struct ggml_tensor * dense_bias;
};

struct rnn_generator load_model(){
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context *ctx = ggml_init(params);
    struct rnn_generator model;

    model.ctx = ctx;

    model.embeddings = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 256, 66);
    model.cell_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3072, 256);
    model.cell_recurrent_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3072, 1024);
    model.cell_bias = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3072, 2);
    model.dense_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 66, 1024);
    model.dense_bias = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 66, 1);

    auto fin = ifstream("rnn_text_gen/gru.bin", ios::binary);

    if (! fin) cout << "Error reading file" << endl;
    int dummy;
    
    // Read the embeddings
    for (int i = 0; i < 3; i++) {
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.embeddings->data, 256*66*sizeof(float));

    // Read the cell kernel
    for (int i = 0; i < 3; i++) {
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.cell_kernel->data, 3072*256*sizeof(float));

    // Read the cell recurrent kernel
    for (int i = 0; i < 3; i++) {
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.cell_recurrent_kernel->data, 3072*1024*sizeof(float));

    // Read the cell bias
    for (int i = 0; i < 3; i++) {
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.cell_bias->data, 3072*2*sizeof(float));

    // Read the dense kernel
    for (int i = 0; i < 3; i++) {
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.dense_kernel->data, 66*1024*sizeof(float));


    // Read the dense kernel
    for (int i = 0; i < 2; i++){
        fin.read((char*)&dummy, sizeof(dummy));
        cout << dummy << endl;
    }
    fin.read((char*)model.dense_bias->data, 66*1*sizeof(float));

    print_2d(model.dense_bias);
    
    fin.close();

    return model;
}

int main(){

    struct rnn_generator my_model = load_model();
}
