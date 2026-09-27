// dlr_c_api.c
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef void* DLRModelHandle;

extern int CreateDLRModel(DLRModelHandle* handle, const char* model_path,
                          int dev_type, int dev_id);
extern int DeleteDLRModel(DLRModelHandle* handle);
extern int SetDLRInput(DLRModelHandle* handle, const char* name,
                       const int64_t* shape, void* input, int dim);
extern int RunDLRModel(DLRModelHandle* handle);
extern int GetDLROutput(DLRModelHandle* handle, int index, void* output);
extern int GetDLROutputSizeDim(DLRModelHandle* handle, int index,
                                int64_t* size, int* dim);
extern int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs);
extern int GetDLRInputName(DLRModelHandle* handle, int index, const char** name);
extern int GetDLRInputShape(DLRModelHandle* handle, int index, int64_t* shape);
extern int GetDLRInputSizeDim(DLRModelHandle* handle, int index,
                               int64_t* size, int* dim);

typedef struct {
    DLRModelHandle handle;
    char     input_name[128];  // actual input name from model
    int64_t  input_shape[4];   // NHWC: [1, 224, 224, 3]
    int      input_dims;
    int64_t  input_elems;      // 1*224*224*3 = 150528
    int64_t  output_elems;     // 1000
} TiState;

void* tidl_rt_init(void) {
    TiState* s = (TiState*)calloc(1, sizeof(TiState));
    if (!s) {
        fprintf(stderr, "[dlr_c_api] Failed to allocate TiState\n");
        return NULL;
    }
    return (void*)s;
}

int tidl_rt_load_model(void* handle,
                       const uint8_t* path,
                       size_t path_len,
                       const uint8_t* artifacts_dir,
                       size_t artifacts_dir_len) {
    (void)artifacts_dir;
    (void)artifacts_dir_len;

    TiState* s = (TiState*)handle;

    char path_str[512];
    if (path_len >= sizeof(path_str)) {
        fprintf(stderr, "[dlr_c_api] Model path too long\n");
        return -1;
    }
    memcpy(path_str, path, path_len);
    path_str[path_len] = '\0';

    // dev_type=1 (CPU), dev_id=0
    if (CreateDLRModel(&s->handle, path_str, 1, 0) != 0) {
        fprintf(stderr, "[dlr_c_api] CreateDLRModel failed for: %s\n", path_str);
        return -1;
    }

    // Get actual input name from model
    const char* name = NULL;
    if (GetDLRInputName(&s->handle, 0, &name) == 0 && name != NULL) {
        strncpy(s->input_name, name, sizeof(s->input_name) - 1);
        s->input_name[sizeof(s->input_name) - 1] = '\0';
        fprintf(stderr, "[dlr_c_api] Input name: %s\n", s->input_name);
    } else {
        // Fallback to known name for TVM-CL-3090-mobileNetV2-tv
        strncpy(s->input_name, "input.1Net_IN", sizeof(s->input_name) - 1);
        fprintf(stderr, "[dlr_c_api] Using fallback input name: %s\n", s->input_name);
    }

    // Get actual input shape — model uses NHWC uint8
    int64_t in_size = 0;
    int     in_dim  = 0;
    if (GetDLRInputSizeDim(&s->handle, 0, &in_size, &in_dim) == 0 && in_dim == 4) {
        GetDLRInputShape(&s->handle, 0, s->input_shape);
        s->input_dims  = in_dim;
        s->input_elems = in_size;
        fprintf(stderr, "[dlr_c_api] Input shape: [%lld,%lld,%lld,%lld]\n",
                (long long)s->input_shape[0], (long long)s->input_shape[1],
                (long long)s->input_shape[2], (long long)s->input_shape[3]);
    } else {
        // Fallback: NHWC [1,224,224,3] = 150528 uint8 elements
        s->input_dims     = 4;
        s->input_shape[0] = 1;
        s->input_shape[1] = 224;
        s->input_shape[2] = 224;
        s->input_shape[3] = 3;
        s->input_elems    = 150528;
        fprintf(stderr, "[dlr_c_api] Using fallback input shape [1,224,224,3]\n");
    }

    // Get output size
    int out_dims = 0;
    if (GetDLROutputSizeDim(&s->handle, 0, &s->output_elems, &out_dims) != 0) {
        s->output_elems = 1000;
        fprintf(stderr, "[dlr_c_api] Using fallback output_elems=1000\n");
    }

    fprintf(stderr, "[dlr_c_api] Model loaded OK. input_elems=%lld output_elems=%lld\n",
            (long long)s->input_elems, (long long)s->output_elems);
    return 0;
}

int tidl_rt_alloc_tensors(void* handle) {
    (void)handle;
    return 0;
}

// input:  uint8 NHWC bytes  [1,224,224,3]
// output: float32 bytes     [1,1000]
int tidl_rt_process(void* handle,
                    const uint8_t* input,  size_t input_bytes,
                    uint8_t*       output, size_t output_bytes) {
    TiState* s = (TiState*)handle;
    (void)input_bytes;
    (void)output_bytes;

    if (SetDLRInput(&s->handle, s->input_name, s->input_shape,
                    (void*)input, s->input_dims) != 0) {
        fprintf(stderr, "[dlr_c_api] SetDLRInput failed (name=%s)\n", s->input_name);
        return -1;
    }

    if (RunDLRModel(&s->handle) != 0) {
        fprintf(stderr, "[dlr_c_api] RunDLRModel failed\n");
        return -2;
    }

    if (GetDLROutput(&s->handle, 0, (void*)output) != 0) {
        fprintf(stderr, "[dlr_c_api] GetDLROutput failed\n");
        return -3;
    }

    return 0;
}

int32_t tidl_rt_get_input_elems(void* handle) {
    TiState* s = (TiState*)handle;
    return (int32_t)s->input_elems;
}

int32_t tidl_rt_get_output_elems(void* handle) {
    TiState* s = (TiState*)handle;
    return (int32_t)s->output_elems;
}

void tidl_rt_destroy(void* handle) {
    TiState* s = (TiState*)handle;
    if (s) {
        DeleteDLRModel(&s->handle);
        free(s);
    }
}

// ── CPU variant (device_type=0) ───────────────────────────────────────────────
void* tidl_rt_init_cpu(void) {
    TiState* s = (TiState*)calloc(1, sizeof(TiState));
    if (!s) return NULL;
    return (void*)s;
}

int tidl_rt_load_model_cpu(void* handle, const uint8_t* path, size_t path_len) {
    TiState* s = (TiState*)handle;
    char path_str[512];
    if (path_len >= sizeof(path_str)) return -1;
    memcpy(path_str, path, path_len);
    path_str[path_len] = '\0';

    // device_type=1 → kDLCPU (ARM Cortex-A72)
    if (CreateDLRModel(&s->handle, path_str, 1, 0) != 0) {
        fprintf(stderr, "[dlr_c_api] CPU CreateDLRModel failed: %s\n", path_str);
        return -1;
    }

    s->input_dims    = 4;
    s->input_shape[0] = 1; s->input_shape[1] = 3;
    s->input_shape[2] = 224; s->input_shape[3] = 224;
    s->input_elems   = 1 * 3 * 224 * 224;
    s->output_elems  = 1000;

    fprintf(stderr, "[dlr_c_api] CPU model loaded OK\n");
    return 0;
}
