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

