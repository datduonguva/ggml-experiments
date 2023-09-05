#include <ggml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void read_csv(string file_name, vector<int> &labels, vector<vector<int>> &pixels){
    fstream input_file;
    input_file.open(file_name, ios::in);


    if (input_file.is_open()){
	cout << "The file is open" << endl;
        string sa;
    
	while(getline(input_file, sa)){
	    
	    int start_index = 0;
		
	    int label;
	    vector<int> pixel; 
	    vector<int> sep_position {0};

	    // get positions of the commas
	    for(int i=0; i<sa.length(); i++)
		if (sa[i] == ',') sep_position.push_back(i);
	    sep_position.push_back(sa.length());

	    //
	    label = stoi(sa.substr(sep_position[0], sep_position[1] - sep_position[0]));


	    for (int i = 1; i < sep_position.size() - 1; i++){
		// the number start as comma's position + 1
		    pixel.push_back(stoi(sa.substr(sep_position[i] + 1, sep_position[i+1] - sep_position[i])));
	    }
		// the number start as comma's position + 1

	    labels.push_back(label);
	    pixels.push_back(pixel);
	}
    }
}
float frand() {
    return (float)rand()/(float)RAND_MAX;
}

void randomize_tensor(
    struct ggml_tensor * tensor,
    int ndims,
    int64_t ne[]
){
    if (ndims == 1){
	for (int i = 0; i < ne[0]; i++){
	    ((float*) tensor->data)[i] = frand();
	}
    } else if (ndims == 2){
	for (int i1 = 0; i1 < ne[1]; i1++){
	    for (int i0 = 0; i0 < ne[0]; i0++){
		((float*) tensor ->data)[i0*ne[1] +i1] = frand() - 0.5f;
	    }
	}
    }
}

void shuffle(vector<int> &arr){
    for (int i = 0; i < arr.size(); i++){
	int new_pos = (int) (frand()*arr.size());
	int tmp = arr[i];
	arr[i] = arr[new_pos];
	arr[new_pos] = tmp;
    }  
}
