OBJ=./objects
GGML=/home/datduong/github/ggml
GGML=/home/duongquocdat7411/github/ggml

all: mkdir main

main: ${OBJ}/ggml.o ${OBJ}/mobilevit.o ${OBJ}/ggml-quants.o 
	g++ -g -o  main ${OBJ}/ggml.o  ${OBJ}/mobilevit.o ${OBJ}/ggml-quants.o  -lm -lpthread

mkdir: 
	ls .. && mkdir -p ${OBJ}


${OBJ}/ggml.o: ${GGML}/src/ggml.c
	gcc -D_GNU_SOURCE -c -I${GGML}/include/ggml  -o ${OBJ}/ggml.o ${GGML}/src/ggml.c  

${OBJ}/ggml-quants.o: ${GGML}/src/ggml-quants.c
	gcc -c -D_GNU_SOURCE -I${GGML}/include/ggml  -o ${OBJ}/ggml-quants.o ${GGML}/src/ggml-quants.c  

${OBJ}/mobilevit.o: main.cpp 
	g++ -std=c++11 -I${GGML}/include -I${GGML}/examples/ -I.. -o ${OBJ}/mobilevit.o -c main.cpp 


.PHONY: clean


clean:
	rm -f ${OBJ}/*.o	
