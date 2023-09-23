OBJ=./objects


main: mkdir ${OBJ}/ggml.o  ${OBJ}/test-mul-mat3.o
	g++ -o main ${OBJ}/test-mul-mat3.o  ${OBJ}/ggml.o -lm -lpthread

mkdir: 
	mkdir -p ${OBJ}

${OBJ}/ggml.o: ../src/ggml.c
	gcc -c -I ../include/ggml/ -o ${OBJ}/ggml.o ../src/ggml.c

${OBJ}/test-simplernn.o: test-simplernn.cpp
	g++ -I../include/ggml/ -o ${OBJ}/test-simplernn.o -c test-simplernn.cpp


${OBJ}/test-mul-mat3.o: test-mul-mat3.cpp
	g++ -I../include/ggml/ -o ${OBJ}/test-mul-mat3.o -c test-mul-mat3.cpp


