#include <ggml.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <cstring>

using namespace std;
/**
Layer: my_model/embedding/embeddings:0 with shape (66, 256)
Layer: my_model/gru/gru_cell/kernel:0 with shape (256, 3072) ->read to 3072, 256 -> transpose 256, 3072
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


// for matrix a (n, m) -> slicing (a[start_index: end_index, :]
struct ggml_tensor * slice_2d(ggml_context * ctx, struct ggml_tensor * t, int start_index, int end_index){
    struct ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, end_index - start_index); 
    for (int i = 0; i < end_index - start_index; i++){
        ggml_set_i32_1d(mask, i, (int32_t) (start_index + i));
    }

    struct ggml_tensor * result = ggml_get_rows(ctx, t, mask); 
    return result;
}

struct ggml_struct * sigmoid_2d(ggml_context * ctx, struct ggml_struct *t){
    int size = t->ne[0]*t->ne[1];

    struct ggml_struct * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
    for(int i = 0; i < size; i++)
        ggml_set_f32_1d(
            result,
            i,
            1.0f/(1.0f + exp(-ggml_get_f32_1d(t, i)))
        );
    return result;
}

int map_char_to_id(char c, map<char, int> char2id){
    auto it = char2id.find(c);
    if (it != char2id.end()){
        return it->second;
    }
    // return the default value
    return char2id['\t'];
}

vector<int> encode_text(string text, map<char, int> &char2id){
    // input has shape [sequence_length, 1] as we are supporting batch of 1
    vector<int> result;
    for (char a: text) result.push_back(map_char_to_id(a, char2id));
    return result;
}


struct rnn_generator {
    struct ggml_context * ctx;
    struct ggml_tensor * embeddings;
    struct ggml_tensor * cell_kernel;
    struct ggml_tensor * cell_recurrent_kernel;
    struct ggml_tensor * input_bias;
    struct ggml_tensor * recurrent_bias;
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
    model.cell_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 256, 3072);
    model.cell_recurrent_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 1024, 3072); 

    model.input_bias = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3072, 1);
    model.recurrent_bias = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 3072, 1);

    model.dense_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 66, 1024);
    model.dense_bias = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 66, 1);

    auto fin = ifstream("rnn_text_gen/gru.bin", ios::binary);

    if (! fin) cout << "Error reading file" << endl;
    int dummy;
    
    // Read the embeddings
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.embeddings->data, 256*66*sizeof(float));

    // Read the cell kernel
    for (int i = 0; i < 3; i++) fin.read((char*)&dummy, sizeof(dummy)); 

    //this is how we saved data, transpose later
    struct ggml_tensor * kernel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3072, 256); 
    fin.read((char*)kernel->data, 3072*256*sizeof(float));
    model.cell_kernel = ggml_cont(ctx, ggml_transpose(ctx, kernel));    // 256, 3072

    // Read the cell recurrent kernel, saved as python [1024, 3072]
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    struct ggml_tensor * recurrent_kernel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3072, 1024);
    fin.read((char*)recurrent_kernel->data, 3072*1024*sizeof(float));
    // chage to 1024, 3072
    model.cell_recurrent_kernel = ggml_transpose(ctx, recurrent_kernel);

    // Read the cell bias
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    struct ggml_tensor * cell_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3072, 2);
    fin.read((char*)cell_bias->data, 3072*2*sizeof(float));
    model.input_bias = slice_2d(ctx, cell_bias, 0, 1);
    model.recurrent_bias = slice_2d(ctx, cell_bias, 1, 2);

    // Read the dense kernel
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.dense_kernel->data, 66*1024*sizeof(float));


    // Read the dense kernel
    for (int i = 0; i < 2; i++){ fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.dense_bias->data, 66*1*sizeof(float));

    fin.close();

    return model;
}


map<char, int> load_char2id(){
    //struct rnn_generator my_model = load_model();
    string a = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    map<char, int> char2id;
    // use \t for unknown
    char2id['\t'] = 0;
    for (int i = 0; i < a.length();i ++){
        char2id[a[i]] = i + 1;
    }

    return char2id;
}

void inference(
    const rnn_generator & model,
    vector<int> & encoded_input
){
    // create the context holding the variables 
    struct ggml_init_params params = {
        .mem_size = 128 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context * ctx0 = ggml_init(params);

    int n_units = model.cell_kernel->ne[1] / 3;
    int embedding_size = model.embeddings->ne[0];
    
    // get the weights from the model

    struct ggml_tensor * states = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_units, 1);      // (1024, 1)
    struct ggml_tensor * input_vector = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, embedding_size, 1);


    struct ggml_tensor * matrix_x;
    struct ggml_tensor * xz, *xr, *xh,* inner_matrix;
    struct ggml_tensor *rz, *rr, *rh, *hh, *z, *r, *h;
    for (int i = 0; i < 100; i++){
        // for c in the prompt
        int char_index;
        if (i < encoded_input.size()){
            // embedding lookup. this step is verified
            char_index = encoded_input.at(i); 
            memcpy(
                (float*) input_vector->data,
                (float*) model.embeddings->data + embedding_size*char_index,
                embedding_size*sizeof(float)
            );
        
            cout << "before matmul: " << endl;
            
            matrix_x = ggml_transpose(ctx0,
                ggml_add(ctx0,
                    ggml_mul_mat(ctx0,
                        model.cell_kernel,
                        input_vector
                    ),
                    model.input_bias
                )
            );          // 3072, 1 -> 1, 3072

            xz = slice_2d(ctx0, matrix_x, 0, 1024);
            xr = slice_2d(ctx0, matrix_x, 1024, 2048);
            xh = slice_2d(ctx0, matrix_x, 2048, 3072);

            inner_matrix = ggml_transpose(ctx0, 
                ggml_add(ctx0,    // 3072, 1
                    ggml_mul_mat(ctx0,           // 3072, 1
                        model.recurrent_kernel, states         // 1024, 3072 , 1024, 1 
                    ),
                    model.recurrent_bias
                )
            );
            rz = slice_2d(ctx0, inner_matrix, 0, 1024);
            rr = slice_2d(ctx0, inner_matrix, 1024, 2048);
            rh = slice_2d(ctx0, inner_matrix, 2048, 3072);

            z = sigmoid_2d(ctx0, ggml_add(ctx0, xz, rz));
            r = sigmoid_2d(ctx0, ggml_add(ctx0, xr, rr));

            rh = ggml_mul(ctx0, r, rh); 
            hh = sigmoid_2d(ctx0, xh, rh);

            h = ggml_add(ctx0,
                ggml_mul(ctx0, z, states),
                ggml_mul(ctx0, 
                    ggml_sub(ctx0,
                        // TODO: I am here
                        1, z
                    ),
                    hh
                )
            );

        }
    }

    struct ggml_cgraph gf = ggml_build_forward(xz);
    ggml_graph_compute_with_ctx(ctx0, &gf, 1);
    print_2d(ggml_transpose(ctx0, xz));
    cout << "xz: " << xz->ne[0] << ", " << xz->ne[1] << endl;
}

int main(){
    
    // load the tokenizer map
    map<char, int> char2id = load_char2id();

    // load the model
    struct rnn_generator my_model = load_model();

    // run the inference code
    string input_prompt = "Take 1 tablet a day for 10 days";
    auto encoded_input = encode_text(input_prompt, char2id); 
    inference(my_model, encoded_input);
}
