OBJ=./objects
GGML=/home/datduong/github/ggml
#GGML=/home/datduong/github/ggml
GGML=/Users/dduong/Repos/github/ggml

all: mkdir main

main: ${OBJ}/ggml.o ${OBJ}/svm.o  ${OBJ}/ggml-quants.o 
	g++ -g -O3 -DGGML_METAL=ON -march=native -o  main ${OBJ}/ggml.o  ${OBJ}/svm.o ${OBJ}/ggml-quants.o -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}

${OBJ}/ggml-quants.o: ${GGML}/src/ggml-quants.c
	gcc  -O3 -c -DGGML_CLBLAST=ON  -D_GNU_SOURCE -march=native -I${GGML}/include/ggml  -o ${OBJ}/ggml-quants.o ${GGML}/src/ggml-quants.c

${OBJ}/ggml.o: ${GGML}/src/ggml.c
	gcc -O3 -DGGML_CLBLAST=ON -D_GNU_SOURCE -march=native -c -I${GGML}/include/ggml  -o ${OBJ}/ggml.o ${GGML}/src/ggml.c  

${OBJ}/svm.o: svm.cpp 
	g++ -O3 -std=c++11  -DGGML_CLBLAST=ON -march=native  -I${GGML}/include -I${GGML}/examples/ -I.. -o ${OBJ}/svm.o -c svm.cpp 


.PHONY: clean


clean:
	rm -f ${OBJ}/*.o	
