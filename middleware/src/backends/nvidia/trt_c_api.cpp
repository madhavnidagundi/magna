// =============================================================================
// Magna Middleware — TensorRT C API Wrapper
// =============================================================================
// Thin C-compatible API around TensorRT for FFI from Rust.
//
// IMPORTANT: This is ONLY compiled when the `nvidia` feature is enabled.
// The build.rs feature-gates this compilation.

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <memory>

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// CUDA Error Checking Macro (H10 fix)
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[TRT-C++] CUDA error in " << #call << ": "       \
                      << cudaGetErrorString(err) << std::endl;              \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_PTR(call)                                                \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "[TRT-C++] CUDA error in " << #call << ": "       \
                      << cudaGetErrorString(err) << std::endl;              \
            return nullptr;                                                 \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// TRT Logger
// ---------------------------------------------------------------------------

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT-C++] " << msg << std::endl;
        }
    }
} gLogger;

// ---------------------------------------------------------------------------
// INT8 Calibration Cache Calibrator
// ---------------------------------------------------------------------------
// Cache-only IInt8EntropyCalibrator2 implementation.
// Reads a pre-computed calibration table from disk; does NOT run calibration
// passes over training data.  This is the standard pattern when the cache has
// already been generated offline (e.g. via `trtexec --int8 --calib=<dataset>`).
// TensorRT requires the calibrator object even when only reading from cache.

class CacheCalibrator : public IInt8EntropyCalibrator2 {
public:
    explicit CacheCalibrator(const std::string& cache_path)
        : cache_path_(cache_path) {
        // Pre-load the cache file into memory
        std::ifstream f(cache_path_, std::ios::binary);
        if (f.good()) {
            f.seekg(0, f.end);
            size_t size = f.tellg();
            f.seekg(0, f.beg);
            cache_data_.resize(size);
            f.read(cache_data_.data(), size);
            f.close();
            std::cout << "[TRT-C++] Calibration cache loaded: " << cache_path_
                      << " (" << size << " bytes)" << std::endl;
        } else {
            std::cerr << "[TRT-C++] Warning: could not read calibration cache: "
                      << cache_path_ << std::endl;
        }
    }

    // Cache-only: we never provide fresh calibration batches
    int32_t getBatchSize() const noexcept override { return 1; }
    bool getBatch(void* /*bindings*/[], const char* /*names*/[], int32_t /*nbBindings*/) noexcept override {
        return false;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        if (cache_data_.empty()) {
            length = 0;
            return nullptr;
        }
        length = cache_data_.size();
        return cache_data_.data();
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream f(cache_path_, std::ios::binary);
        if (f.good()) {
            f.write(reinterpret_cast<const char*>(cache), length);
            f.close();
            std::cout << "[TRT-C++] Calibration cache written: " << cache_path_
                      << " (" << length << " bytes)" << std::endl;
        } else {
            std::cerr << "[TRT-C++] Warning: could not write calibration cache: "
                      << cache_path_ << std::endl;
        }
    }

private:
    std::string cache_path_;
    std::vector<char> cache_data_;
};

// ---------------------------------------------------------------------------
// TrtContext — holds all TRT and CUDA state
// ---------------------------------------------------------------------------

struct TrtContext {
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* d_input;
    void* d_output;
    int input_size;   // number of input elements
    int output_size;  // number of output elements
    int in_elem_size; // bytes per input element
    int out_elem_size; // bytes per output element
    cudaStream_t stream;
    int in_idx;
    int out_idx;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool has_graph;
};

extern "C" {

// ---------------------------------------------------------------------------
// Load a serialized TRT engine from a file path (with length)
// ---------------------------------------------------------------------------

void* trt_load_engine(const uint8_t* path_ptr, size_t path_len) {
    // Construct a null-terminated string from the Rust &[u8]
    std::string path(reinterpret_cast<const char*>(path_ptr), path_len);

    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[TRT-C++] Engine file not found: " << path << std::endl;
        return nullptr;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "[TRT-C++] Failed to create InferRuntime" << std::endl;
        return nullptr;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
    if (!engine) {
        std::cerr << "[TRT-C++] Failed to deserialize engine" << std::endl;
        delete runtime;
        return nullptr;
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "[TRT-C++] Failed to create execution context" << std::endl;
        delete engine;
        delete runtime;
        return nullptr;
    }

    TrtContext* ctx = new TrtContext();
    ctx->runtime = runtime;
    ctx->engine = engine;
    ctx->context = context;

    // Discover I/O tensor indices and shapes dynamically
    ctx->in_idx = -1;
    ctx->out_idx = -1;
    const char* in_name = nullptr;
    const char* out_name = nullptr;

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            ctx->in_idx = i;
            in_name = name;
        } else if (engine->getTensorIOMode(name) == TensorIOMode::kOUTPUT) {
            ctx->out_idx = i;
            out_name = name;
        }
    }

    if (ctx->in_idx < 0 || ctx->out_idx < 0) {
        std::cerr << "[TRT-C++] Engine has no input/output tensors" << std::endl;
        delete context;
        delete engine;
        delete runtime;
        delete ctx;
        return nullptr;
    }

    // Read shapes dynamically from the engine (no hardcoded 224×224)
    auto in_dims = engine->getTensorShape(in_name);
    auto out_dims = engine->getTensorShape(out_name);

    ctx->input_size = 1;
    for (int i = 0; i < in_dims.nbDims; i++) {
        if (in_dims.d[i] > 0) ctx->input_size *= in_dims.d[i];
    }

    ctx->output_size = 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        if (out_dims.d[i] > 0) ctx->output_size *= out_dims.d[i];
    }

    // Determine element sizes based on data type
    ctx->in_elem_size = 4; // default FP32
    DataType in_type = engine->getTensorDataType(in_name);
    if (in_type == DataType::kHALF) ctx->in_elem_size = 2;
    else if (in_type == DataType::kINT8) ctx->in_elem_size = 1;

    ctx->out_elem_size = 4; // default FP32
    DataType out_type = engine->getTensorDataType(out_name);
    if (out_type == DataType::kHALF) ctx->out_elem_size = 2;
    else if (out_type == DataType::kINT8) ctx->out_elem_size = 1;

    // Allocate device memory with error checking (H10 fix)
    CUDA_CHECK_PTR(cudaMalloc(&ctx->d_input, ctx->input_size * ctx->in_elem_size));
    CUDA_CHECK_PTR(cudaMalloc(&ctx->d_output, ctx->output_size * ctx->out_elem_size));
    ctx->stream = nullptr;
    ctx->graph = nullptr;
    ctx->graph_exec = nullptr;
    ctx->has_graph = false;
    CUDA_CHECK_PTR(cudaStreamCreate(&ctx->stream));

    std::cout << "[TRT-C++] Engine loaded: input=" << ctx->input_size
              << " output=" << ctx->output_size
              << " in_elem=" << ctx->in_elem_size
              << " out_elem=" << ctx->out_elem_size << std::endl;

    return (void*)ctx;
}

// ---------------------------------------------------------------------------
// Allocate buffers (called separately from Rust)
// ---------------------------------------------------------------------------

int trt_allocate_buffers(void* handle) {
    // Buffers are already allocated in trt_load_engine.
    // This exists for API symmetry with the Rust InferenceBackend trait.
    if (!handle) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// Destroy engine and free all resources
// ---------------------------------------------------------------------------

void trt_destroy_engine(void* handle) {
    if (!handle) return;
    TrtContext* ctx = (TrtContext*)handle;

    if (ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
        cudaFree(ctx->d_input);
        cudaFree(ctx->d_output);
        if (ctx->graph_exec) cudaGraphExecDestroy(ctx->graph_exec);
        if (ctx->graph) cudaGraphDestroy(ctx->graph);
        cudaStreamDestroy(ctx->stream);
    }

    if (ctx->context) delete ctx->context;
    if (ctx->engine) delete ctx->engine;
    if (ctx->runtime) delete ctx->runtime;

    delete ctx;
    std::cout << "[TRT-C++] Engine destroyed" << std::endl;
}

// ---------------------------------------------------------------------------
// Query tensor sizes
// ---------------------------------------------------------------------------

int trt_get_input_elems(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->input_size;
}

int trt_get_output_elems(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->output_size;
}

int trt_get_input_elem_size(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->in_elem_size;
}

int trt_get_output_elem_size(void* handle) {
    if (!handle) return 0;
    return ((TrtContext*)handle)->out_elem_size;
}

// ---------------------------------------------------------------------------
// Run inference with CUDA error checking
// ---------------------------------------------------------------------------

int trt_infer(void* handle,
              const uint8_t* input, size_t input_bytes,
              uint8_t* output, size_t output_bytes) {
    if (!handle) return -1;
    TrtContext* ctx = (TrtContext*)handle;

    // Validate sizes
    size_t expected_in = (size_t)ctx->input_size * ctx->in_elem_size;
    if (input_bytes != expected_in && input_bytes != (size_t)ctx->input_size * 4) {
        std::cerr << "[TRT-C++] Input size mismatch: got " << input_bytes
                  << " expected " << expected_in << std::endl;
        return -3;
    }

    // Copy input to device
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_input, input, input_bytes,
                               cudaMemcpyHostToDevice, ctx->stream));

    // Set tensor addresses
    const char* in_name = ctx->engine->getIOTensorName(ctx->in_idx);
    const char* out_name = ctx->engine->getIOTensorName(ctx->out_idx);
    ctx->context->setTensorAddress(in_name, ctx->d_input);
    ctx->context->setTensorAddress(out_name, ctx->d_output);

    // If the input shape is dynamic, we must set it explicitly before enqueue
    auto in_dims = ctx->engine->getTensorShape(in_name);
    bool dynamic = false;
    for (int i = 0; i < in_dims.nbDims; i++) {
        if (in_dims.d[i] == -1) {
            in_dims.d[i] = 1; // Fallback to batch 1
            dynamic = true;
        }
    }
    if (dynamic) {
        ctx->context->setInputShape(in_name, in_dims);
    }

    // Execute
    bool status = ctx->context->enqueueV3(ctx->stream);
    if (!status) {
        std::cerr << "[TRT-C++] enqueueV3 failed" << std::endl;
        return -2;
    }

    // Copy output back to host
    size_t copy_bytes = std::min(output_bytes,
                                (size_t)ctx->output_size * ctx->out_elem_size);
    CUDA_CHECK(cudaMemcpyAsync(output, ctx->d_output, copy_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    return 0;
}

extern "C" bool trt_capture_graph(void* handle) {
    auto* ctx = static_cast<TrtContext*>(handle);
    if (!ctx || !ctx->context) return false;

    // Capture the graph on the context's stream
    CUDA_CHECK(cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal));
    
    // Set addresses (must be consistent for graph)
    const char* in_name = ctx->engine->getIOTensorName(ctx->in_idx);
    const char* out_name = ctx->engine->getIOTensorName(ctx->out_idx);
    ctx->context->setTensorAddress(in_name, ctx->d_input);
    ctx->context->setTensorAddress(out_name, ctx->d_output);

    // Enqueue
    ctx->context->enqueueV3(ctx->stream);

    CUDA_CHECK(cudaStreamEndCapture(ctx->stream, &ctx->graph));
    CUDA_CHECK(cudaGraphInstantiate(&ctx->graph_exec, ctx->graph, 0));
    
    ctx->has_graph = true;
    return true;
}

extern "C" int trt_infer_graph(void* handle, const void* input, size_t input_bytes, void* output, size_t output_bytes) {
    auto* ctx = static_cast<TrtContext*>(handle);
    if (!ctx || !ctx->has_graph) return -1;

    // Copy to device (still async)
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_input, input, input_bytes, cudaMemcpyHostToDevice, ctx->stream));

    // Launch captured graph
    CUDA_CHECK(cudaGraphLaunch(ctx->graph_exec, ctx->stream));

    // Copy back
    size_t copy_bytes = std::min(output_bytes, (size_t)ctx->output_size * ctx->out_elem_size);
    CUDA_CHECK(cudaMemcpyAsync(output, ctx->d_output, copy_bytes, cudaMemcpyDeviceToHost, ctx->stream));
    
    // Sync for safety in this synchronous API
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    return 0;
}

// ---------------------------------------------------------------------------
// Get device memory usage (using actual element sizes, not just sizeof(float))
// ---------------------------------------------------------------------------

long long trt_get_device_memory(void* handle) {
    if (!handle) return 0;
    TrtContext* ctx = (TrtContext*)handle;

    long long total = (long long)ctx->engine->getDeviceMemorySize();
    // Use actual element sizes, not sizeof(float) (M8 fix)
    total += (long long)ctx->input_size * ctx->in_elem_size;
    total += (long long)ctx->output_size * ctx->out_elem_size;
    return total;
}

// == Engine Builder =========================================================

extern "C" bool trt_build_engine(
    const char* onnx_path,
    const char* engine_path,
    int precision, // 0: FP32, 1: FP16, 2: INT8, 3: FP8
    const char* calib_cache
) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) return false;

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) return false;

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return false;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnx_path, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        return false;
    }

    // Precision configuration
    if (precision >= 1) { // FP16
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }

    // Calibrator instance — must outlive buildSerializedNetwork() call
    std::unique_ptr<CacheCalibrator> calibrator;

    if (precision == 2) { // INT8
        if (builder->platformHasFastInt8()) {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);

            if (calib_cache && std::strlen(calib_cache) > 0) {
                // Validate calibration cache file exists
                std::ifstream cache_check(calib_cache);
                if (!cache_check.good()) {
                    std::cerr << "[TRT-C++] Calibration cache file not found: "
                              << calib_cache << std::endl;
                    return false;
                }
                cache_check.close();

                calibrator = std::make_unique<CacheCalibrator>(calib_cache);
                config->setInt8Calibrator(calibrator.get());
                std::cout << "[TRT-C++] INT8 calibrator attached (cache: "
                          << calib_cache << ")" << std::endl;
            } else {
                std::cout << "[TRT-C++] WARNING: INT8 precision requested without "
                          << "calibration cache — using TensorRT layer-wise fallback"
                          << std::endl;
            }
        }
    }

    if (precision == 3) { // FP8 (Thor / Blackwell specific)
        #if NV_TENSORRT_MAJOR >= 9
        config->setFlag(nvinfer1::BuilderFlag::kFP8);
        std::cout << "[TRT-C++] Enabling FP8 optimization for Thor hardware" << std::endl;
        
        // Thor-specific: Increase workspace for Blackwell's larger L2
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 32); // 4GB
        
        // Thor-specific: Limit tactics to fast modern paths, but retain JIT and EdgeMask for Depthwise convs
        config->setTacticSources(1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS) |
                                 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT) |
                                 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUDNN) |
                                 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS) |
                                 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kJIT_CONVOLUTIONS));
        #endif
    }


    // Optimization Profile for dynamic shapes
    auto profile = builder->createOptimizationProfile();
    bool hasDynamic = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto input = network->getInput(i);
        auto dims = input->getDimensions();
        bool isInputDynamic = false;
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) isInputDynamic = true;
        }

        if (isInputDynamic) {
            hasDynamic = true;
            nvinfer1::Dims minDims = dims;
            nvinfer1::Dims optDims = dims;
            nvinfer1::Dims maxDims = dims;

            for (int j = 0; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    if (j == 0) { // Batch
                        minDims.d[j] = 1; optDims.d[j] = 1; maxDims.d[j] = 1;
                    } else if (j == 1 && dims.nbDims == 4) { // Channels
                        minDims.d[j] = 3; optDims.d[j] = 3; maxDims.d[j] = 3;
                    } else { // Height/Width
                        minDims.d[j] = 224; optDims.d[j] = 224; maxDims.d[j] = 224;
                    }
                }
            }
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims);
        }
    }
    if (hasDynamic) {
        config->addOptimizationProfile(profile);
    }

    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) return false;

    std::ofstream outfile(engine_path, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "[TRT-C++] Failed to open output file: " << engine_path << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    outfile.close();
    if (!outfile.good()) {
        std::cerr << "[TRT-C++] Failed to write engine to: " << engine_path << std::endl;
        return false;
    }

    return true;
}

} // extern "C"
