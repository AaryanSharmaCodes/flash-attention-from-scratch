NVCC   ?= nvcc
# -arch=native asks nvcc to target whatever card is actually present, which
# avoids the JIT step and makes the timings reflect real compiled code.
# Override on older toolkits: make ARCH=sm_75
ARCH   ?= native
# -lineinfo costs nothing at runtime and lets Nsight Compute map counters back
# to source lines, which is the whole point of profiling this.
NVCCFLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH) -Iinclude
CXXFLAGS  := -O2 -std=c++17

BUILD := build

.PHONY: all bench test clean
all: $(BUILD)/bench_sgemm $(BUILD)/bench_softmax $(BUILD)/test_reference

$(BUILD):
	@mkdir -p $(BUILD) results

$(BUILD)/bench_sgemm: src/host/bench_sgemm.cu src/kernels/sgemm.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcublas

$(BUILD)/bench_softmax: src/host/bench_softmax.cu src/kernels/softmax.cu ref/reference.cpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# The reference test is plain C++ on purpose, so it builds and runs on a
# machine with no GPU at all.
$(BUILD)/test_reference: tests/test_reference.cpp ref/reference.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^

bench: $(BUILD)/bench_sgemm $(BUILD)/bench_softmax
	./$(BUILD)/bench_sgemm 1024
	./$(BUILD)/bench_sgemm 2048
	./$(BUILD)/bench_sgemm 4096
	./$(BUILD)/bench_softmax 4096 1024
	./$(BUILD)/bench_softmax 4096 4096

test: $(BUILD)/test_reference
	./$(BUILD)/test_reference

clean:
	rm -rf $(BUILD)
