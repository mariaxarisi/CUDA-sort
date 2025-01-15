# Compiler and flags
NVCC = nvcc
CFLAGS = -O2

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Source files
COMMON_SRC_FILES = $(SRC_DIR)/vector.c
SRC_FILE_QSORT = $(SRC_DIR)/qsort.c
SRC_FILE_V0 = $(SRC_DIR)/V0.cu
SRC_FILE_V1 = $(SRC_DIR)/V1.cu
SRC_FILE_V2 = $(SRC_DIR)/V2.cu
MAIN_FILE = $(SRC_DIR)/main.c

# Object files
COMMON_OBJ_FILES = $(BUILD_DIR)/vector.obj
MAIN_OBJ_QSORT = $(BUILD_DIR)/main_qsort.obj
MAIN_OBJ_V0 = $(BUILD_DIR)/main_v0.obj
MAIN_OBJ_V1 = $(BUILD_DIR)/main_v1.obj
MAIN_OBJ_V2 = $(BUILD_DIR)/main_v2.obj
OBJ_FILE_QSORT = $(BUILD_DIR)/qsort.obj
OBJ_FILE_V0 = $(BUILD_DIR)/V0.obj
OBJ_FILE_V1 = $(BUILD_DIR)/V1.obj
OBJ_FILE_V2 = $(BUILD_DIR)/V2.obj

# Executable names
EXEC_QSORT = $(BIN_DIR)/qsort
EXEC_V0 = $(BIN_DIR)/V0
EXEC_V1 = $(BIN_DIR)/V1
EXEC_V2 = $(BIN_DIR)/V2

# Targets
.PHONY: all clean run

all: $(BIN_DIR) $(BUILD_DIR) $(EXEC_V0) $(EXEC_V1) $(EXEC_V2) $(EXEC_QSORT)

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

$(EXEC_QSORT): $(MAIN_OBJ_QSORT) $(COMMON_OBJ_FILES) $(OBJ_FILE_QSORT)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) $^ -o $@

$(EXEC_V0): $(MAIN_OBJ_V0) $(COMMON_OBJ_FILES) $(OBJ_FILE_V0)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) $^ -o $@

$(EXEC_V1): $(MAIN_OBJ_V1) $(COMMON_OBJ_FILES) $(OBJ_FILE_V1)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) $^ -o $@

$(EXEC_V2): $(MAIN_OBJ_V2) $(COMMON_OBJ_FILES) $(OBJ_FILE_V2)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) $^ -o $@

$(BUILD_DIR)/vector.obj: $(SRC_DIR)/vector.c
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(MAIN_OBJ_QSORT): $(MAIN_FILE)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -DVERSION_QSORT -c $< -o $@

$(MAIN_OBJ_V0): $(MAIN_FILE)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -DVERSION_V0 -c $< -o $@

$(MAIN_OBJ_V1): $(MAIN_FILE)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -DVERSION_V1 -c $< -o $@

$(MAIN_OBJ_V2): $(MAIN_FILE)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -DVERSION_V2 -c $< -o $@

$(BUILD_DIR)/qsort.obj: $(SRC_FILE_QSORT)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/V0.obj: $(SRC_FILE_V0)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/V1.obj: $(SRC_FILE_V1)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/V2.obj: $(SRC_FILE_V2)
	$(NVCC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

run: all
	$(EXEC_QSORT) $(filter-out $@,$(MAKECMDGOALS))
	$(EXEC_V0) $(filter-out $@,$(MAKECMDGOALS))
	$(EXEC_V1) $(filter-out $@,$(MAKECMDGOALS))
	$(EXEC_V2) $(filter-out $@,$(MAKECMDGOALS))
	
%:
	@: