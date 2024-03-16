#include <ggml.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <map>
#include <cstring>
#include <cmath>

using namespace std;
/**
Layer: my_model/embedding/embeddings:0 with shape (66, 256)
Layer: my_model/gru/gru_cell/kernel:0 with shape (256, 3072) ->read to 3072, 256 -> transpose 256, 3072
Layer: my_model/gru/gru_cell/recurrent_kernel:0 with shape (1024, 3072)
Layer: my_model/gru/gru_cell/bias:0 with shape (2, 3072)
Layer: my_model/dense/kernel:0 with shape (1024, 66)
Layer: my_model/dense/bias:0 with shape (66,)
**/

string vocab = "\t\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
void print_2d(ggml_tensor * ts){
    for (int i1 = 0; i1 < ts->ne[1]; i1++){
        for (int i0 = 0; i0 < ts->ne[0]; i0++){
            if (i0 == 10) break;
            cout << ggml_get_f32_1d(ts, i1*ts->ne[0] + i0) << " ";    
        }
        cout << endl;
    }   
}

void print_shape(struct ggml_tensor * t, string name){
    cout << name << " ";
    for (int i = 0; i < t->n_dims; i++)
        cout << t->ne[i] << " ";
    cout << endl;
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

struct ggml_tensor * sigmoid_2d(ggml_context * ctx, struct ggml_tensor * x) {
    // ggml has no native sigmoid, but silu(x) / x can be an approximation
    x = ggml_div(ctx, ggml_silu(ctx, x), x);
    return x;
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
    for (int i = 0; i< text.length(); i++) result.push_back(map_char_to_id(text[i], char2id));
    return result;
}

// max index on the first dimension
int argmax_1d(struct ggml_tensor * t){
    float * probs = ggml_get_data_f32(t);     
    return max_element(probs, probs + t->ne[0]) - probs;
}

struct rnn_generator {
    struct ggml_context * ctx;
    struct ggml_tensor * embeddings;
    struct ggml_tensor * cell_kernel;
    struct ggml_tensor * cell_kernel_t;
    struct ggml_tensor * cell_recurrent_kernel;
    struct ggml_tensor * cell_recurrent_kernel_t;
    struct ggml_tensor * cell_bias;
    struct ggml_tensor * dense_kernel;
    struct ggml_tensor * dense_kernel_t;
    struct ggml_tensor * dense_bias;
};

void print_text(vector<int> ids){
    for(int i = 0; i< ids.size(); i++) cout << vocab[ids[i]] ;
    cout << endl;
    cout << "--------" << endl;
}
struct rnn_generator load_model(){
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
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
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.embeddings->data, 256*66*sizeof(float));

    // Read the cell kernel
    for (int i = 0; i < 3; i++) fin.read((char*)&dummy, sizeof(dummy)); 
    fin.read((char*)model.cell_kernel->data, 3072*256*sizeof(float));
    model.cell_kernel_t = ggml_cont(ctx, ggml_transpose(ctx, model.cell_kernel));

    // Read the cell recurrent kernel, saved as python [1024, 3072]
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*) model.cell_recurrent_kernel->data, 3072*1024*sizeof(float));
    model.cell_recurrent_kernel_t = ggml_cont(ctx, ggml_transpose(ctx, model.cell_recurrent_kernel));

    // Read the cell bias
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.cell_bias->data, 3072*2*sizeof(float));

    // Read the dense kernel
    for (int i = 0; i < 3; i++) { fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.dense_kernel->data, 66*1024*sizeof(float));
    model.dense_kernel_t = ggml_cont(ctx, ggml_transpose(ctx, model.dense_kernel));

    // Read the dense kernel
    for (int i = 0; i < 2; i++){ fin.read((char*)&dummy, sizeof(dummy)); }
    fin.read((char*)model.dense_bias->data, 66*1*sizeof(float));

    struct ggml_cgraph gf = {}; 
    ggml_set_param(ctx, model.cell_kernel);
    ggml_set_param(ctx, model.cell_recurrent_kernel);
    ggml_set_param(ctx, model.dense_kernel);
 
    ggml_build_forward_expand(&gf, model.cell_kernel_t);
    ggml_build_forward_expand(&gf, model.cell_recurrent_kernel_t);
    ggml_build_forward_expand(&gf, model.dense_kernel_t);

    ggml_graph_compute_with_ctx(ctx, &gf, 1);

    print_2d(model.dense_kernel_t);
    fin.close();
    return model;

}


map<char, int> load_char2id(){
    map<char, int> char2id;
    for (int i = 0; i < vocab.length();i ++){
        char2id[vocab[i]] = i;
    }

    return char2id;
}

struct cell_output {
    struct ggml_tensor * output;
    struct ggml_tensor * states;
    struct ggml_tensor * z;
    struct ggml_tensor * matrix_x;

};



struct cell_output gru_forward(
    struct ggml_context * ctx0,
    struct rnn_generator &  model,
    struct ggml_tensor * input_id,
    struct ggml_tensor * states
){
    struct cell_output cell;

    int embedding_size = 256;
    struct ggml_tensor * input_vector;
    struct ggml_tensor * matrix_x, *output;
    struct ggml_tensor * xz, *xr, *xh,* inner_matrix;
    struct ggml_tensor *rz, *rr, *rh, *hh, *z, *r, *h;

    input_vector = ggml_cont(ctx0, ggml_get_rows(ctx0, model.embeddings, input_id));

    matrix_x = ggml_transpose(ctx0,
        ggml_add(ctx0,
            ggml_mul_mat(ctx0,
                model.cell_kernel_t, //3072, 256 -> 256, 3072
                input_vector
            ),
            slice_2d(ctx0, model.cell_bias, 0, 1)
        )
    );          // 3072, 1 -> 1, 3072

    matrix_x->nb[0] = sizeof(float);
    xz = slice_2d(ctx0, matrix_x, 0, 1024);         // 1, 1024
    xr = slice_2d(ctx0, matrix_x, 1024, 2048);      // 1, 1024
    xh = slice_2d(ctx0, matrix_x, 2048, 3072);      // 1, 1024

    inner_matrix = ggml_transpose(ctx0, 
        ggml_add(ctx0,    // 3072, 1
            ggml_mul_mat(ctx0,           // 3072, 1
                model.cell_recurrent_kernel_t, // 3072, 1024 -> 
                states         // 1024, 3072 , 1024, 1 
            ),
            slice_2d(ctx0, model.cell_bias, 1, 2) 
        )
    );
    inner_matrix->nb[0] = sizeof(float);
    rz = slice_2d(ctx0, inner_matrix, 0, 1024);
    rr = slice_2d(ctx0, inner_matrix, 1024, 2048);
    rh = slice_2d(ctx0, inner_matrix, 2048, 3072);

    z = sigmoid_2d(ctx0, ggml_add(ctx0, xz, rz)); // (1, 1024)
    r = sigmoid_2d(ctx0, ggml_add(ctx0, xr, rr));


    rh = ggml_mul(ctx0, r, rh);         // (1, 1024)
    hh = ggml_tanh(ctx0, ggml_add(ctx0, xh, rh)); // (1, 1024)


    states = ggml_transpose(ctx0,       //(1024, 1)
        ggml_add(ctx0,
            ggml_mul(ctx0, z, ggml_transpose(ctx0, states)),
            ggml_mul(ctx0, 
                ggml_sub(ctx0,
                    ggml_repeat(ctx0, ggml_new_f32(ctx0, 1.0f), z),
                    z
                ),
                hh
            )
        )
    );
    
    output = ggml_add(ctx0,
        ggml_mul_mat(ctx0, 
            model.dense_kernel_t,
            states
        ),
        model.dense_bias
    );
    cell.output = output;
    cell.states = states;

    return cell;
}
    

void inference(
    struct rnn_generator & model,
    vector<int> & encoded_input
){
    // create the context holding the variables 
    struct ggml_init_params params = {
        .mem_size = 1 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };
    struct ggml_context * ctx0 = ggml_init(params);

    int n_units = model.cell_kernel->ne[0] / 3;
    int embedding_size = model.embeddings->ne[0];
    
    // get the weights from the model

    struct ggml_tensor * states = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_units, 1);      // (1024, 1)
    struct ggml_tensor * input_id = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    struct cell_output  output = gru_forward(ctx0, model, input_id, states);
    ggml_set_param(ctx0, states);
    ggml_set_param(ctx0, input_id);

    struct ggml_cgraph gf = ggml_build_forward(output.output);

    // feed the graph with input, get the output
    int char_index;
    for (int i = 0; i < 200; i++){
        // for c in the prompt
        if (i < encoded_input.size()){
            // embedding lookup. this step is verified
            char_index = encoded_input[i];
        } else{
            encoded_input.push_back(char_index);
            print_text(encoded_input);
        }

        ggml_set_i32_1d(input_id, 0, char_index);
        if (i != 0){
            memcpy(
                (float*) states->data,
                (float*) output.states->data,
                1024*sizeof(float)
            );
        }
        ggml_graph_compute_with_ctx(ctx0, &gf, 1);
        char_index = argmax_1d(output.output);
    }
}

int main(){
    
    // load the tokenizer map
    map<char, int> char2id = load_char2id();

    // load the model
    struct rnn_generator my_model = load_model();

    // run the inference code
    char input_prompt[50];
    cout << "type: " << endl;
    cin.getline(input_prompt, 50);
    auto encoded_input = encode_text(input_prompt, char2id); 
    cout << endl;
    inference(my_model, encoded_input);
}
