#include <ggml.h>
#include <iostream>
#include <vector>
#include <cstring>

using namespace std; 

void print_2d(ggml_tensor * ts){
    for (int i1 = 0; i1 < ts->ne[1]; i1++){
        for (int i0 = 0; i0 < ts->ne[0]; i0++){
            cout << ggml_get_f32_1d(ts, i1*ts->ne[0] + i0) << " ";    
        }
        cout << endl;
    }   
}

struct simple_rnn {
    struct ggml_context * ctx;
    struct ggml_tensor * kernel;
    struct ggml_tensor * recurrent_kernel;
};


struct simple_rnn load_simple_rnn(){
    struct ggml_init_params params = {
        .mem_size = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };

    struct ggml_context * ctx = ggml_init(params);

    struct simple_rnn model;
    model.ctx = ctx;
    model.kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 8, 4);
    model.recurrent_kernel = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 4, 4);

    float kernel[] =  {
        0.6469684, 0.18856752, -0.523959, -0.5145968, 0.0648613, -0.655183,
        0.26202983, -0.47914273, -0.058191, 0.15540725, 0.14797556, -0.54640806,
        -0.06038815, 0.15216482, -0.48145998, -0.11173499, 0.31529063, 0.11578298,
        -0.13875079, 0.64682513, 0.48433453, 0.22317564, -0.3239754, 0.09182703,
        -0.41143027, -0.6579423, 0.6903445, -0.44512737, -0.21940672, -0.5509994,
        0.06007087, 0.12950546};

    float  recurrent_kernel[] = {
        -0.6053593, 0.68987226, -0.2725365, -0.28868738, 0.15599209, -0.1458273,
        0.2767996, -0.9369007, 0.57175523, 0.12090871, -0.79578954, -0.15873256,
        -0.5313216, -0.6987072, -0.46456954, -0.11696438};

    // Set the values:
    for(int i1 = 0; i1 < 4; i1++){
        for (int i0 = 0 ; i0 < 8; i0 ++){
            ggml_set_f32_1d(model.kernel, i1*8 + i0, kernel[i0*4 + i1]);
        }

        for (int i0 = 0; i0 < 4; i0 ++){
            ggml_set_f32_1d(model.recurrent_kernel, i1*4 + i0, recurrent_kernel[i0*4+i1]);
        }
    }

    return model;
}

int main(){
    struct ggml_init_params params = {
        .mem_size = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc = false
    };

    float np_input[] = {
       0.37884507, 0.07736231, 0.99417198, 0.55363131, 0.36397761,
       0.12024814, 0.2770327 , 0.93440062, 0.50849497, 0.56501603,
       0.5087595 , 0.83295339, 0.71214545, 0.88972557, 0.2724725 ,
       0.83435857, 0.75211012, 0.85677063, 0.60132718, 0.08080171,
       0.04330026, 0.19639964, 0.60163999, 0.70150995, 0.26080886,
       0.15800025, 0.50137049, 0.50872868, 0.78667116, 0.19347423,
       0.20229633, 0.92455524, 0.14410123, 0.81341422, 0.7685774 ,
       0.03808706, 0.32736173, 0.21933064, 0.6023525 , 0.14633271,
       0.24288404, 0.84350854, 0.08868112, 0.1516823 , 0.80128568,
       0.93519557, 0.00484885, 0.39033198};

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 3, 2); 
    for (int i0 = 0; i0 < 8; i0++){
        for (int i1 = 0; i1 < 3; i1 ++){
            for(int i2 = 0; i2 < 2; i2++){
                ggml_set_f32_1d(input, i2*8*3 + i1*8 + i0, np_input[ i2*8*3 + i1*8 + i0]);
            }
        }
    }


    struct simple_rnn model =  load_simple_rnn();

    struct ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);   
    for (int step = 0; step < 3; step ++){
        struct ggml_tensor * input_step = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);   

        // Copy over the data, this should be used as a slicing function
        for (int batch = 0; batch < 2; batch++){
            memcpy(
                (float*) input_step->data + 8*batch,
                (float*) input->data + batch*24 + step*8 + 0,
                8*sizeof(float));
        }

        struct ggml_tensor * h = ggml_mul_mat(ctx, model.kernel, input_step);  // (4, 2)
        struct ggml_tensor * h1 = ggml_mul_mat(ctx, model.recurrent_kernel, output); // (4, 2)
        struct ggml_tensor * h2   = ggml_add(ctx, h, h1);

        output = ggml_tanh(ctx, h2);
    }
    
    struct ggml_cgraph gf = ggml_build_forward(output); 
    ggml_graph_compute_with_ctx(ctx, &gf, 1);

    cout << "kernel" << endl;
    print_2d(model.kernel);
    cout << "\nrecurrent_kernel" << endl;
    print_2d(model.recurrent_kernel);
    cout << "\noutput" << endl;
    print_2d(output);
    

    float expected_output[] = {-0.37519965, -0.57683633,  0.31615578, -0.8848848 , 0.89704936, -0.07763906, -0.54329058,  0.5651122};

    struct ggml_tensor * expected_ts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);

    memcpy(expected_ts->data, expected_output, 8*sizeof(float));

    cout << "\nExpected output: " << endl;
    print_2d(expected_ts);
    return 0;
}
