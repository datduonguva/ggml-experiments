OBJ=./objects
GGML=/home/datduong/github/ggml
#GGML=/home/datduong/github/ggml

all: mkdir main

main: ${OBJ}/ggml.o ${OBJ}/mobilevit.o ${OBJ}/ggml-quants.o 
	g++ -g -O3 -march=native -o  main ${OBJ}/ggml.o  ${OBJ}/mobilevit.o ${OBJ}/ggml-quants.o  -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}

${OBJ}/ggml.o: ${GGML}/src/ggml.c
	gcc -O3 -D_GNU_SOURCE -march=native -c -I${GGML}/include/ggml  -o ${OBJ}/ggml.o ${GGML}/src/ggml.c  

${OBJ}/ggml-quants.o: ${GGML}/src/ggml-quants.c
	gcc  -O3 -c -D_GNU_SOURCE -march=native -I${GGML}/include/ggml  -o ${OBJ}/ggml-quants.o ${GGML}/src/ggml-quants.c  

${OBJ}/mobilevit.o: main.cpp 
	g++ -O3 -std=c++11  -DGGML_CLBLAST=ON  -I${GGML}/include -I${GGML}/examples/ -I.. -o ${OBJ}/mobilevit.o -c main.cpp 


.PHONY: clean


clean:
	rm -f ${OBJ}/*.o	
