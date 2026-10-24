NVCC ?= nvcc
NVCCFLAGS ?= -O2

BIN_DIR := build/bin
OBJ_DIR := build/obj

.PHONY: all demos kernels clean

all: demos kernels

demos: $(BIN_DIR)/vectorAdd $(BIN_DIR)/vectorMul $(BIN_DIR)/saxpy $(BIN_DIR)/test

kernels: $(OBJ_DIR)/sumReduce.o $(OBJ_DIR)/softmax.o $(OBJ_DIR)/swiGLU.o \
	$(OBJ_DIR)/gemm.o $(OBJ_DIR)/rope.o $(OBJ_DIR)/onlineSoftmax.o

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

$(BIN_DIR)/vectorAdd: vectorAdd.cu cuda_header.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I. $< -o $@

$(BIN_DIR)/vectorMul: vectorMul.cu cuda_header.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I. $< -o $@

$(BIN_DIR)/saxpy: saxpy.cu cuda_header.cuh | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I. $< -o $@

$(BIN_DIR)/test: test.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(OBJ_DIR)/sumReduce.o: sumReduce.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

$(OBJ_DIR)/softmax.o: softmax.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

$(OBJ_DIR)/swiGLU.o: swiGLU.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

$(OBJ_DIR)/gemm.o: gemm.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

$(OBJ_DIR)/rope.o: rope.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

$(OBJ_DIR)/onlineSoftmax.o: onlineSoftmax.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I. -c $< -o $@

clean:
	rm -rf build
