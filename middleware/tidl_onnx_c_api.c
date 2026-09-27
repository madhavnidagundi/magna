// tidl_onnx_c_api.c
//
// TI TDA4 backend shim that drives the ONNX Runtime TIDL Execution Provider
// directly through the ONNX Runtime C API. Used on hosts (or boards) that
// ship a TIDL-enabled `libonnxruntime.so` but not the DLR runtime.
//
// Exposes the same tidl_rt_* surface that the DLR shim (dlr_c_api.c) exposes,
// so backends/ti/adapter.rs can call either implementation interchangeably.
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "onnxruntime/core/session/onnxruntime_c_api.h"

// -----------------------------------------------------------------------
// TIDL execution provider C API (declared locally instead of including
// onnxruntime/core/providers/tidl/tidl_provider_factory.h, whose bare
// `#include "onnxruntime_c_api.h"` doesn't resolve against this source
// tree's include layout). Struct layout and symbol names must match
// exactly what libonnxruntime.so exports.
//
// Note: this library's generic OrtApi::SessionOptionsAppendExecutionProvider
// (string-keyed provider options, used by python's `providers=[("TIDLExecutionProvider",
// {...})]`) only recognizes SNPE/XNNPACK/AZURE here — TIDL is only reachable
// through this dedicated struct-based entry point. `OrtSessionsOptionsSetDefault_Tidl`
// must be used to populate sane defaults for priority/max_pre_empt_delay/core_number;
// zero-initializing them (as if `core_number=0` were "auto") causes
// TIDL_createStateInferFunc to fail with -16.
// -----------------------------------------------------------------------
typedef struct {
    int debug_level;
    char artifacts_folder[512];
    int priority;
    float max_pre_empt_delay;
    int core_number;
} c_api_tidl_options;

extern OrtStatus* OrtSessionsOptionsSetDefault_Tidl(c_api_tidl_options* tidl_options);
extern OrtStatus* OrtSessionOptionsAppendExecutionProvider_Tidl(OrtSessionOptions* options,
                                                                 c_api_tidl_options* tidl_options);

#define MAX_NAME_LEN 256
#define MAX_DIMS 8

typedef struct {
    const OrtApi* api;
    OrtEnv* env;
    OrtSession* session;
    OrtMemoryInfo* mem_info;

    char input_name[MAX_NAME_LEN];
    char output_name[MAX_NAME_LEN];

    int64_t input_shape[MAX_DIMS];
    size_t input_dims;
    int64_t input_elems;

    int64_t output_shape[MAX_DIMS];
    size_t output_dims;
    int64_t output_elems;
} TidlOnnxState;

static void log_status(const OrtApi* api, const char* where, OrtStatus* status) {
    if (status) {
        fprintf(stderr, "[tidl_onnx] %s failed: %s\n", where, api->GetErrorMessage(status));
    }
}

void* tidl_rt_init(void) {
    TidlOnnxState* s = (TidlOnnxState*)calloc(1, sizeof(TidlOnnxState));
    if (!s) {
        fprintf(stderr, "[tidl_onnx] Failed to allocate state\n");
        return NULL;
    }

    s->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!s->api) {
        fprintf(stderr, "[tidl_onnx] OrtGetApiBase()->GetApi failed (version mismatch?)\n");
        free(s);
        return NULL;
    }

    OrtStatus* status = s->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "magna-ti-onnx", &s->env);
    if (status) {
        log_status(s->api, "CreateEnv", status);
        s->api->ReleaseStatus(status);
        free(s);
        return NULL;
    }

    return (void*)s;
}

// Populates s->input_name/input_shape/input_dims/input_elems from the
// currently-loaded session's input #0 (and same for output #0).
static int query_io_metadata(TidlOnnxState* s) {
    const OrtApi* api = s->api;
    OrtAllocator* allocator = NULL;
    OrtStatus* status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        log_status(api, "GetAllocatorWithDefaultOptions", status);
        api->ReleaseStatus(status);
        return -1;
    }

    // --- Input name ---
    char* in_name = NULL;
    status = api->SessionGetInputName(s->session, 0, allocator, &in_name);
    if (status) {
        log_status(api, "SessionGetInputName", status);
        api->ReleaseStatus(status);
        return -1;
    }
    strncpy(s->input_name, in_name, sizeof(s->input_name) - 1);
    allocator->Free(allocator, in_name);

    // --- Input shape ---
    OrtTypeInfo* in_type_info = NULL;
    status = api->SessionGetInputTypeInfo(s->session, 0, &in_type_info);
    if (status) {
        log_status(api, "SessionGetInputTypeInfo", status);
        api->ReleaseStatus(status);
        return -1;
    }
    const OrtTensorTypeAndShapeInfo* in_tensor_info = NULL;
    api->CastTypeInfoToTensorInfo(in_type_info, &in_tensor_info);
    size_t in_dims = 0;
    api->GetDimensionsCount(in_tensor_info, &in_dims);
    if (in_dims > MAX_DIMS) in_dims = MAX_DIMS;
    api->GetDimensions(in_tensor_info, s->input_shape, in_dims);
    s->input_dims = in_dims;
    api->ReleaseTypeInfo(in_type_info);

    s->input_elems = 1;
    for (size_t i = 0; i < in_dims; i++) {
        int64_t d = s->input_shape[i];
        if (d <= 0) {
            d = 1; // dynamic batch/dim -> assume 1
            s->input_shape[i] = 1;
        }
        s->input_elems *= d;
    }

    // --- Output name ---
    char* out_name = NULL;
    status = api->SessionGetOutputName(s->session, 0, allocator, &out_name);
    if (status) {
        log_status(api, "SessionGetOutputName", status);
        api->ReleaseStatus(status);
        return -1;
    }
    strncpy(s->output_name, out_name, sizeof(s->output_name) - 1);
    allocator->Free(allocator, out_name);

    // --- Output shape ---
    OrtTypeInfo* out_type_info = NULL;
    status = api->SessionGetOutputTypeInfo(s->session, 0, &out_type_info);
    if (status) {
        log_status(api, "SessionGetOutputTypeInfo", status);
        api->ReleaseStatus(status);
        return -1;
    }
    const OrtTensorTypeAndShapeInfo* out_tensor_info = NULL;
    api->CastTypeInfoToTensorInfo(out_type_info, &out_tensor_info);
    size_t out_dims = 0;
    api->GetDimensionsCount(out_tensor_info, &out_dims);
    if (out_dims > MAX_DIMS) out_dims = MAX_DIMS;
    api->GetDimensions(out_tensor_info, s->output_shape, out_dims);
    s->output_dims = out_dims;
    api->ReleaseTypeInfo(out_type_info);

    s->output_elems = 1;
    for (size_t i = 0; i < out_dims; i++) {
        int64_t d = s->output_shape[i];
        if (d <= 0) {
            d = 1;
            s->output_shape[i] = 1;
        }
        s->output_elems *= d;
    }

    fprintf(stderr,
            "[tidl_onnx] input='%s' elems=%lld | output='%s' elems=%lld\n",
            s->input_name, (long long)s->input_elems, s->output_name,
            (long long)s->output_elems);
    return 0;
}

// model_path / artifacts_dir are passed as separate (non-NUL-terminated)
// byte slices from Rust; lengths are explicit.
int tidl_rt_load_model(void* handle, const uint8_t* model_path, size_t model_path_len,
                        const uint8_t* artifacts_dir, size_t artifacts_dir_len) {
    TidlOnnxState* s = (TidlOnnxState*)handle;
    const OrtApi* api = s->api;

    char model_path_str[1024];
    char artifacts_str[512];

    if (model_path_len >= sizeof(model_path_str)) {
        fprintf(stderr, "[tidl_onnx] Model path too long\n");
        return -1;
    }
    memcpy(model_path_str, model_path, model_path_len);
    model_path_str[model_path_len] = '\0';

    if (artifacts_dir_len >= sizeof(artifacts_str)) {
        fprintf(stderr, "[tidl_onnx] Artifacts dir too long\n");
        return -1;
    }
    memcpy(artifacts_str, artifacts_dir, artifacts_dir_len);
    artifacts_str[artifacts_dir_len] = '\0';

    OrtSessionOptions* session_options = NULL;
    OrtStatus* status = api->CreateSessionOptions(&session_options);
    if (status) {
        log_status(api, "CreateSessionOptions", status);
        api->ReleaseStatus(status);
        return -1;
    }

    c_api_tidl_options tidl_opts;
    memset(&tidl_opts, 0, sizeof(tidl_opts));
    status = OrtSessionsOptionsSetDefault_Tidl(&tidl_opts);
    if (status) {
        log_status(api, "OrtSessionsOptionsSetDefault_Tidl", status);
        api->ReleaseStatus(status);
        api->ReleaseSessionOptions(session_options);
        return -2;
    }
    strncpy(tidl_opts.artifacts_folder, artifacts_str, sizeof(tidl_opts.artifacts_folder) - 1);
    const char* debug_level_env = getenv("MAGNA_TIDL_DEBUG_LEVEL");
    tidl_opts.debug_level = debug_level_env ? atoi(debug_level_env) : 0;

    fprintf(stderr,
            "[tidl_onnx] Loading model='%s' artifacts_folder='%s' priority=%d "
            "max_pre_empt_delay=%f core_number=%d\n",
            model_path_str, artifacts_str, tidl_opts.priority, tidl_opts.max_pre_empt_delay,
            tidl_opts.core_number);

    status = OrtSessionOptionsAppendExecutionProvider_Tidl(session_options, &tidl_opts);
    if (status) {
        log_status(api, "OrtSessionOptionsAppendExecutionProvider_Tidl", status);
        api->ReleaseStatus(status);
        api->ReleaseSessionOptions(session_options);
        return -2;
    }

    status = api->CreateSession(s->env, model_path_str, session_options, &s->session);
    api->ReleaseSessionOptions(session_options);
    if (status) {
        log_status(api, "CreateSession", status);
        api->ReleaseStatus(status);
        return -3;
    }

    if (query_io_metadata(s) != 0) {
        return -4;
    }

    status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &s->mem_info);
    if (status) {
        log_status(api, "CreateCpuMemoryInfo", status);
        api->ReleaseStatus(status);
        return -5;
    }

    fprintf(stderr, "[tidl_onnx] Model loaded OK\n");
    return 0;
}

int tidl_rt_alloc_tensors(void* handle) {
    (void)handle;
    return 0;
}

// input:  float32 bytes, `input_bytes` long, laid out per the queried input shape.
// output: float32 bytes, caller-allocated buffer of `output_bytes` bytes.
int tidl_rt_process(void* handle, const uint8_t* input, size_t input_bytes, uint8_t* output,
                     size_t output_bytes) {
    TidlOnnxState* s = (TidlOnnxState*)handle;
    const OrtApi* api = s->api;

    OrtValue* input_tensor = NULL;
    OrtStatus* status = api->CreateTensorWithDataAsOrtValue(
        s->mem_info, (void*)input, input_bytes, s->input_shape, s->input_dims,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) {
        log_status(api, "CreateTensorWithDataAsOrtValue", status);
        api->ReleaseStatus(status);
        return -1;
    }

    const char* input_names[1] = {s->input_name};
    const char* output_names[1] = {s->output_name};
    const OrtValue* inputs[1] = {input_tensor};
    OrtValue* outputs[1] = {NULL};

    status = api->Run(s->session, NULL, input_names, inputs, 1, output_names, 1, outputs);
    api->ReleaseValue(input_tensor);
    if (status) {
        log_status(api, "Run", status);
        api->ReleaseStatus(status);
        return -2;
    }

    float* out_data = NULL;
    status = api->GetTensorMutableData(outputs[0], (void**)&out_data);
    if (status) {
        log_status(api, "GetTensorMutableData", status);
        api->ReleaseStatus(status);
        api->ReleaseValue(outputs[0]);
        return -3;
    }

    size_t copy_bytes = (size_t)s->output_elems * sizeof(float);
    if (copy_bytes > output_bytes) copy_bytes = output_bytes;
    memcpy(output, out_data, copy_bytes);

    api->ReleaseValue(outputs[0]);
    return 0;
}

int32_t tidl_rt_get_input_elems(void* handle) {
    return (int32_t)((TidlOnnxState*)handle)->input_elems;
}

int32_t tidl_rt_get_output_elems(void* handle) {
    return (int32_t)((TidlOnnxState*)handle)->output_elems;
}

void tidl_rt_destroy(void* handle) {
    TidlOnnxState* s = (TidlOnnxState*)handle;
    if (!s) return;
    const OrtApi* api = s->api;
    if (api) {
        if (s->mem_info) api->ReleaseMemoryInfo(s->mem_info);
        if (s->session) api->ReleaseSession(s->session);
        if (s->env) api->ReleaseEnv(s->env);
    }
    free(s);
}
