MobileViT

# To build the codes:

* go to mobilevit directory
```
cd mobilevit
```

* pull the ggml repository

```
git clone https://github.com/ggerganov/ggml
```

* generate the weights from TFMobileViTModel:
```
python convert-tf-to-ggml.py
```

This should create a file name "weight.ggml"

* edit the Makefile to point the variable GGML to the location where the ggml's repo is cloned to
```
GGML=/path/to/ggml_repo_dir
```

* build the code 
```
make
```

* run the code to generate the image's features extracted by this MobileViT model:
```
./main
```


expected output:
```
output feature shape: : Dims: (8, 8, 640)
features of the test image: 
i0 = 0, i1 = 0
3.48242, 4.40234, 4.73047, 1.98438, 3.5293, ...4.39844, 4.08203, 2.45703, 2.98828, 3.67578, 
```
