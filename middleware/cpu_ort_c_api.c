#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static const OrtApi *api = NULL;
static OrtEnv *env = NULL;

int cpu_ort_init(void)
{
    const OrtApiBase *base = OrtGetApiBase();
    if (!base) {
        fprintf(stderr, "cpu_ort_init: OrtGetApiBase failed\n");
        return -1;
    }

    api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "cpu_ort_init: GetApi failed\n");
        return -2;
    }

    OrtStatus *st = api->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING,
        "magna_cpu",
        &env);

    if (st) {
        fprintf(stderr, "cpu_ort_init: CreateEnv failed\n");
        api->ReleaseStatus(st);
        return -3;
    }

    return 0;
}

static OrtSession *g_session = NULL;

int cpu_ort_load_model(const char *model_path)
{
    if (!api || !env) {
        fprintf(stderr, "cpu_ort_load_model: not initialized (call cpu_ort_init first)\n");
        return -1;
    }

    OrtSessionOptions *session_options = NULL;
    OrtStatus *st = api->CreateSessionOptions(&session_options);
    if (st) {
        fprintf(stderr, "cpu_ort_load_model: CreateSessionOptions failed\n");
        api->ReleaseStatus(st);
        return -2;
    }

    st = api->CreateSession(env, model_path, session_options, &g_session);
    api->ReleaseSessionOptions(session_options);

    if (st) {
        const char *msg = api->GetErrorMessage(st);
        fprintf(stderr, "cpu_ort_load_model: CreateSession failed: %s\n", msg ? msg : "unknown");
        api->ReleaseStatus(st);
        return -3;
    }

    fprintf(stderr, "cpu_ort_load_model: session created OK\n");
    return 0;
}

void cpu_ort_destroy(void)
{
    if (g_session && api) {
        api->ReleaseSession(g_session);
        g_session = NULL;
    }
    if (env && api) {
        api->ReleaseEnv(env);
        env = NULL;
    }
}

static OrtAllocator *g_allocator = NULL;

static int ensure_allocator(void)
{
    if (g_allocator) return 0;
    OrtStatus *st = api->GetAllocatorWithDefaultOptions(&g_allocator);
    if (st) {
        fprintf(stderr, "ensure_allocator: failed\n");
        api->ReleaseStatus(st);
        return -1;
    }
    return 0;
}

int cpu_ort_get_input_name(char *buf, int buf_len)
{
    if (!g_session || ensure_allocator() != 0) return -1;

    char *name = NULL;
    OrtStatus *st = api->SessionGetInputName(g_session, 0, g_allocator, &name);
    if (st) {
        fprintf(stderr, "cpu_ort_get_input_name: failed\n");
        api->ReleaseStatus(st);
        return -2;
    }
    snprintf(buf, buf_len, "%s", name);
    g_allocator->Free(g_allocator, name);
    return 0;
}

int cpu_ort_get_output_name(char *buf, int buf_len)
{
    if (!g_session || ensure_allocator() != 0) return -1;

    char *name = NULL;
    OrtStatus *st = api->SessionGetOutputName(g_session, 0, g_allocator, &name);
    if (st) {
        fprintf(stderr, "cpu_ort_get_output_name: failed\n");
        api->ReleaseStatus(st);
        return -2;
    }
    snprintf(buf, buf_len, "%s", name);
    g_allocator->Free(g_allocator, name);
    return 0;
}

/*
 * Runs inference. input_data must be a float32 buffer laid out as
 * NCHW [1,3,224,224] (150528 floats). output_data must be a caller-
 * allocated float32 buffer of at least 1000 floats; actual output
 * element count is written to *output_count.
 *
 * Returns 0 on success, negative on failure.
 */
int cpu_ort_run(const float *input_data, int64_t input_elems,
                 float *output_data, int64_t output_buf_capacity,
                 int64_t *output_count)
{
    if (!g_session || !api) {
        fprintf(stderr, "cpu_ort_run: not initialized\n");
        return -1;
    }

    char input_name[128];
    char output_name[128];
    if (cpu_ort_get_input_name(input_name, sizeof(input_name)) != 0) return -2;
    if (cpu_ort_get_output_name(output_name, sizeof(output_name)) != 0) return -3;

    OrtMemoryInfo *mem_info = NULL;
    OrtStatus *st = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    if (st) {
        fprintf(stderr, "cpu_ort_run: CreateCpuMemoryInfo failed\n");
        api->ReleaseStatus(st);
        return -4;
    }

    int64_t input_shape[4] = {1, 3, 224, 224};
    OrtValue *input_tensor = NULL;
    st = api->CreateTensorWithDataAsOrtValue(
        mem_info,
        (void *)input_data,
        input_elems * sizeof(float),
        input_shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    api->ReleaseMemoryInfo(mem_info);
    if (st) {
        fprintf(stderr, "cpu_ort_run: CreateTensorWithDataAsOrtValue failed\n");
        api->ReleaseStatus(st);
        return -5;
    }

    const char *input_names[1]  = { input_name };
    const char *output_names[1] = { output_name };
    OrtValue *output_tensor = NULL;

    st = api->Run(g_session, NULL,
                   input_names, (const OrtValue *const *)&input_tensor, 1,
                   output_names, 1, &output_tensor);

    api->ReleaseValue(input_tensor);

    if (st) {
        const char *msg = api->GetErrorMessage(st);
        fprintf(stderr, "cpu_ort_run: Run failed: %s\n", msg ? msg : "unknown");
        api->ReleaseStatus(st);
        return -6;
    }

    float *out_data = NULL;
    st = api->GetTensorMutableData(output_tensor, (void **)&out_data);
    if (st) {
        fprintf(stderr, "cpu_ort_run: GetTensorMutableData failed\n");
        api->ReleaseStatus(st);
        api->ReleaseValue(output_tensor);
        return -7;
    }

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    st = api->GetTensorTypeAndShape(output_tensor, &shape_info);
    if (st) {
        api->ReleaseStatus(st);
        api->ReleaseValue(output_tensor);
        return -8;
    }
    size_t elem_count = 0;
    api->GetTensorShapeElementCount(shape_info, &elem_count);
    api->ReleaseTensorTypeAndShapeInfo(shape_info);

    if ((int64_t)elem_count > output_buf_capacity) {
        fprintf(stderr, "cpu_ort_run: output buffer too small (%zu > %ld)\n",
                elem_count, (long)output_buf_capacity);
        api->ReleaseValue(output_tensor);
        return -9;
    }

    memcpy(output_data, out_data, elem_count * sizeof(float));
    *output_count = (int64_t)elem_count;

    api->ReleaseValue(output_tensor);
    return 0;
}
