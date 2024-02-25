#include "ggml/ggml.h"

#include "common.h"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

struct mobilevit_hparams {
    int num_channels = 3;
    int image_size = 256;
    int patch_size = 2;
    int hidden_sizes[3] = {144, 192, 240};
    int neck_hidden_sizes[7] = {16, 32, 64, 96, 128, 160, 640};
    int num_attention_heads = 4;
    float mlp_ratio = 2.0;
    float expand_ratio = 4.0;
    std::string hidden_act = "silu";
    int conv_kernel_size = 3;
    int output_stride = 32;
    float hidden_dropout_prob = 0.1;
    float attention_probs_dropout_prob = 0.0;
    float classifier_dropout_prob = 0.1;
    float initializer_range=0.02;
    float layer_norm_eps = 1e-5;
    bool qkv_bias = true;
};

struct mobilevit_conv_layer {
    struct ggml_tensor * kernel;    
//    struct ggml_tensor * bias;    // this conv layer never use bias
    struct ggml_tensor * gamma;
    struct ggml_tensor * beta;
};

struct inverted_residual_layer {
    int in_channels;
    int out_channels;
    int strides;
    int dilation = 1;
    bool use_residual;
    mobilevit_conv_layer expand_1x1;
    mobilevit_conv_layer conv_3x3;
    mobilevit_conv_layer reduce_1x1;
};

struct mobile_net_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides;
    std::vector<inverted_residual_layer> residual_layers;
};

struct mobilevit_transformer_layer{
    // attention
    ggml_tensor * attention_query_kernel;
    ggml_tensor * attention_query_bias;

    ggml_tensor * attention_key_kernel;
    ggml_tensor * attention_key_bias;

    ggml_tensor * attention_value_kernel;
    ggml_tensor * attention_value_bias;

    ggml_tensor * attention_output_kernel;
    ggml_tensor * attention_output_bias;

    // intermediate
    ggml_tensor * intermediate_kernel;
    ggml_tensor * intermediate_bias;

    // output
    ggml_tensor * output_kernel;
    ggml_tensor * output_bias;

    // layernorm_before
    ggml_tensor * lb_gamma;
    ggml_tensor * lb_beta;

    // layernorm_after
    ggml_tensor * la_gamma;
    ggml_tensor * la_beta;
};

struct mobilevit_transformer {
    int in_channels;
    int out_channels;
    int num_stages;
    int hidden_size;
    std::vector<mobilevit_transformer_layer> layers;
};

struct mobile_vit_layer {
    int in_channels;
    int out_channels;
    int num_stages;
    int strides;
    int hidden_size;
    int dilation;

    inverted_residual_layer downsampling_layer;
    mobilevit_conv_layer  conv_kxk;
    mobilevit_conv_layer conv_1x1;
    mobilevit_transformer transformer;
    
    ggml_tensor * layernorm_alpha;
    ggml_tensor * layernorm_beta;
    mobilevit_conv_layer conv_projection;
    mobilevit_conv_layer fusion;
};

struct mobilevit_encoder {
    mobile_net_layer layer_1;        
    mobile_net_layer layer_2;


    mobile_vit_layer layer_3;
    mobile_vit_layer layer_4;
    mobile_vit_layer layer_5;
};

struct mobilevit_model {
    mobilevit_hparams hparams;

    mobilevit_conv_layer conv_stem; 
    mobilevit_encoder encoder;  

    struct ggml_context * ctx_w;    // context for model's weights
};

void read_weights(ggml_tensor * tensor, ggml_context * ctx_w, std::ifstream &fin){
    int name_length, n_dims;
    // read name_length
    fin.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));

    // read name
    std::string name(name_length, 0);
    fin.read(&name[0], name_length);
    std::cout << "name: " << name << " ";

    // read n_dims
    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    std::cout << "n_dims: " << n_dims << ". ";
    
    int dims[4];
    std::cout << "Dim: (";
    for (int i = 0; i < n_dims; i++){
        fin.read(reinterpret_cast<char *>(&dims[i]), sizeof(int));
        std::cout <<  dims[i];
        if (i == n_dims - 1) std::cout << ")\n"; else std::cout << ", ";
    }
    // read the kernel
    if (n_dims == 4){ 
        tensor = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2], dims[3]);
    }else if (n_dims == 3){
        tensor = ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, dims[0], dims[1], dims[2]);
    }else if (n_dims == 2){
        tensor = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, dims[0], dims[1]);
    }else if (n_dims == 1){
        tensor = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, dims[0]);
    }


    fin.read(
        reinterpret_cast<char *>(tensor->data),
        ggml_nbytes(tensor)
    );
}
 
// overloading read mobileit_conv_layer
void read_weights(
    mobilevit_conv_layer & layer,
    ggml_context * ctx_w,
    std::ifstream &fin,
    bool read_gamma=true,
    bool read_beta=true
){
    read_weights(layer.kernel, ctx_w, fin); 
    if (read_gamma) read_weights(layer.gamma, ctx_w, fin);
    if (read_beta) read_weights(layer.beta, ctx_w, fin);
}

// overloading read transformer layers
void read_weights(
    mobilevit_transformer_layer & layer,
    ggml_context * ctx_w,
    std::ifstream &fin,
    bool read_gamma=true,
    bool read_beta=true
){

    read_weights(layer.attention_query_kernel, ctx_w, fin);
    read_weights(layer.attention_query_bias, ctx_w, fin);

    read_weights(layer.attention_key_kernel, ctx_w, fin); 
    read_weights(layer.attention_key_bias, ctx_w, fin);

    read_weights(layer.attention_value_kernel, ctx_w, fin);
    read_weights(layer.attention_value_bias, ctx_w, fin);

    read_weights(layer.attention_output_kernel, ctx_w, fin);
    read_weights(layer.attention_output_bias, ctx_w, fin);


    // intermediate
    read_weights(layer.intermediate_kernel, ctx_w, fin);
    read_weights(layer.intermediate_kernel, ctx_w, fin);   


    // output
    read_weights(layer.output_kernel, ctx_w, fin);
    read_weights(layer.output_bias, ctx_w, fin);

    // layernorm_before
    read_weights(layer.lb_gamma, ctx_w, fin);
    read_weights(layer.lb_beta, ctx_w, fin);

    // layernorm_after
    read_weights(layer.la_gamma, ctx_w, fin);
    read_weights(layer.la_beta, ctx_w, fin);
}

// overloading the read_weights ofthe mobile_vit_layer
void read_weights(mobile_vit_layer &layer, ggml_context * ctx_w, std::ifstream &fin){
    auto downsampling_layer = layer.downsampling_layer;
    read_weights(downsampling_layer.expand_1x1, ctx_w, fin);
    read_weights(downsampling_layer.conv_3x3, ctx_w, fin);
    read_weights(downsampling_layer.reduce_1x1, ctx_w, fin);
   
    read_weights(layer.conv_kxk, ctx_w, fin);
    // this conv_1x1 doesn't have normalization, so, there is no gamma and betta parameters
    read_weights(layer.conv_1x1, ctx_w, fin, false, false);

    // read the transformer layers:
    layer.transformer.layers.resize(layer.num_stages);
    for (int i = 0; i < layer.num_stages; i++){
        auto transformer_layer = layer.transformer.layers[0]; 
        read_weights(transformer_layer, ctx_w, fin);
    }
    // read the layernorm
    read_weights(layer.layernorm_alpha, ctx_w, fin);
    read_weights(layer.layernorm_beta, ctx_w, fin);

    // read the projection
    read_weights(layer.conv_projection, ctx_w, fin);

    // read the fusion
    read_weights(layer.fusion, ctx_w, fin);

}


void load_model(mobilevit_model & model, std::string model_path){
    auto fin = std::ifstream(model_path, std::ios::binary);
    if (!fin){
        std::cout << "Error opening file" << std::endl;
    }

    // read layer conv_stem
    {
        read_weights(model.conv_stem, model.ctx_w, fin); 
    }


    // read encoder
    {
        // read layer_1
        {
            int in_channels = model.hparams.neck_hidden_sizes[0];
            int out_channels = model.hparams.neck_hidden_sizes[1];
            int strides = 1;
            int num_stages = 1;
            // set the parasm
            model.encoder.layer_1.in_channels = in_channels;
            model.encoder.layer_1.out_channels = out_channels;
            model.encoder.layer_1.num_stages = num_stages;
            model.encoder.layer_1.strides = strides;
            model.encoder.layer_1.residual_layers.resize(num_stages);

            for (int i = 0; i < num_stages; i++){
                auto residual_layer = model.encoder.layer_1.residual_layers[i];
                residual_layer.in_channels = in_channels;
                residual_layer.out_channels = out_channels;
                residual_layer.strides = i == 0 ? strides : 1;
                
                read_weights(residual_layer.expand_1x1, model.ctx_w, fin);
                read_weights(residual_layer.conv_3x3, model.ctx_w, fin);
                read_weights(residual_layer.reduce_1x1, model.ctx_w, fin);

                // after the first residual layer, in_channels equals out_channels
                in_channels = out_channels;
            }

        }        

        // read layer 2
        {
            int in_channels = model.hparams.neck_hidden_sizes[1];
            int out_channels = model.hparams.neck_hidden_sizes[2];
            int strides = 2;
            int num_stages = 3;
            // set the parasm
            model.encoder.layer_2.in_channels = in_channels;
            model.encoder.layer_2.out_channels = out_channels;
            model.encoder.layer_2.num_stages = num_stages;
            model.encoder.layer_2.strides = strides;
            model.encoder.layer_2.residual_layers.resize(num_stages);

            for (int i = 0; i < num_stages; i++){
                auto residual_layer = model.encoder.layer_2.residual_layers[i];
                residual_layer.in_channels = in_channels;
                residual_layer.out_channels = out_channels;
                residual_layer.strides = i == 0 ? strides : 1;
                
                read_weights(residual_layer.expand_1x1, model.ctx_w, fin);
                read_weights(residual_layer.conv_3x3, model.ctx_w, fin);
                read_weights(residual_layer.reduce_1x1, model.ctx_w, fin);
                // after the first residual layer, in_channels equals out_channels
                in_channels = out_channels;
            }
        }

        // read layer_3
        {
            int in_channels = model.hparams.neck_hidden_sizes[2];
            int out_channels = model.hparams.neck_hidden_sizes[3];
            int strides = 2;
            int num_stages = 2;
            int hidden_size = model.hparams.hidden_sizes[0];

            model.encoder.layer_3.in_channels = in_channels;
            model.encoder.layer_3.out_channels = out_channels;
            model.encoder.layer_3.num_stages = num_stages;
            model.encoder.layer_3.strides = strides;
            model.encoder.layer_3.hidden_size = hidden_size;
            model.encoder.layer_3.dilation = 1;

            read_weights(model.encoder.layer_3, model.ctx_w, fin);
        }

        // read layer 4
        {
            int in_channels = model.hparams.neck_hidden_sizes[3];
            int out_channels = model.hparams.neck_hidden_sizes[4];
            int strides = 2;
            int num_stages = 4;
            int hidden_size = model.hparams.hidden_sizes[1];

            model.encoder.layer_4.in_channels = in_channels;
            
            model.encoder.layer_4.out_channels = out_channels;
            model.encoder.layer_4.num_stages = num_stages;
            model.encoder.layer_4.strides = strides;
            model.encoder.layer_4.hidden_size = hidden_size;
            model.encoder.layer_4.dilation = 1;

            read_weights(model.encoder.layer_4, model.ctx_w, fin); 
        }

        // read layer 5
        {
            int in_channels = model.hparams.neck_hidden_sizes[4];
            int out_channels = model.hparams.neck_hidden_sizes[5];
            int strides = 2 ;
            int num_stages = 3;
            int hidden_size = model.hparams.hidden_sizes[2];

            model.encoder.layer_4.in_channels = in_channels;
            
            model.encoder.layer_4.out_channels = out_channels;
            model.encoder.layer_4.num_stages = num_stages;
            model.encoder.layer_4.strides = strides;
            model.encoder.layer_4.hidden_size = hidden_size;
            model.encoder.layer_4.dilation = 1;

            read_weights(model.encoder.layer_4, model.ctx_w, fin); 
        }
    }

    fin.close();
}

int main(int argc, char ** argv) {
    ggml_time_init();
    mobilevit_model model;

    // create ggml_context for model's weight
    {
        struct ggml_init_params params = {128*1024*1024, NULL, false};

        model.ctx_w = ggml_init(params);
        if (!model.ctx_w) {
            std::cout << "Cannot create context for model's weights" << std::endl;
        }
    }

    load_model(model, "weight.ggml");
    return 0;
}
