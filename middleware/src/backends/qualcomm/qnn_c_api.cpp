// =============================================================================
// Magna Middleware — Qualcomm QNN FFI Wrapper
// =============================================================================
// C++ wrapper around Qualcomm's QNN C API. Exposes a simple interface to Rust.

#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include <cstdint>

#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnBackend.h"
#include "QnnGraph.h"
#include "QnnProperty.h"
#include "QnnTensor.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

static thread_local std::string g_last_error;

static void set_last_error(const std::string& message) {
    g_last_error = message;
}

extern "C" {

struct QnnContextHandle {
    void* backend_lib_handle;
    void* system_lib_handle;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;

    Qnn_Tensor_t* inputs;
    uint32_t num_inputs;
    Qnn_Tensor_t* outputs;
    uint32_t num_outputs;
    QnnSystemContext_Handle_t system_context;

    // Core Functions
    QnnBackend_CreateFn_t QnnBackend_create;
    QnnBackend_FreeFn_t QnnBackend_free;
    QnnContext_CreateFromBinaryFn_t QnnContext_createFromBinary;
    QnnContext_FreeFn_t QnnContext_free;
    QnnGraph_RetrieveFn_t QnnGraph_retrieve;
    QnnGraph_ExecuteFn_t QnnGraph_execute;

    // System Functions for metadata
    QnnSystemContext_CreateFn_t QnnSystemContext_create;
    QnnSystemContext_GetMetaDataFn_t QnnSystemContext_getMetadata;
    QnnSystemContext_FreeFn_t QnnSystemContext_free;
};

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

static int tensor_precision(const Qnn_Tensor_t& tensor) {
    Qnn_DataType_t data_type;
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        data_type = tensor.v1.dataType;
    } else if (tensor.version == QNN_TENSOR_VERSION_2) {
        data_type = tensor.v2.dataType;
    } else {
        return -1;
    }

    switch (data_type) {
        case QNN_DATATYPE_FLOAT_32:
            return 0;
        case QNN_DATATYPE_FLOAT_16:
            return 1;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
            return 2;
        default:
            return -1;
    }
}

static const char* tensor_name(const Qnn_Tensor_t& tensor) {
    if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.name;
    if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.name;
    return nullptr;
}

// Expected client-buffer size (bytes) for a tensor: product(dims) * element
// width, where the width is derived from the same coarse precision classes as
// tensor_precision() (fp32->4, fp16->2, 8-bit quantised->1). Returns 0 when it
// can't be determined (unknown version/dtype/dims) so the caller treats the
// size as unvalidatable rather than wrong.
static uint32_t tensor_expected_bytes(const Qnn_Tensor_t& tensor) {
    int prec = tensor_precision(tensor);
    uint32_t width;
    switch (prec) {
        case 0:  width = 4; break;  // FLOAT_32
        case 1:  width = 2; break;  // FLOAT_16
        case 2:  width = 1; break;  // INT_8 / UINT_8 / [US]FIXED_POINT_8
        default: return 0;
    }

    uint32_t rank;
    const uint32_t* dimensions;
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        rank = tensor.v1.rank;
        dimensions = tensor.v1.dimensions;
    } else if (tensor.version == QNN_TENSOR_VERSION_2) {
        rank = tensor.v2.rank;
        dimensions = tensor.v2.dimensions;
    } else {
        return 0;
    }
    if (!dimensions || rank == 0 || rank > 8) return 0;

    uint64_t total = width;
    for (uint32_t i = 0; i < rank; ++i) {
        if (dimensions[i] == 0 || total > UINT32_MAX / dimensions[i]) return 0;
        total *= dimensions[i];
    }
    return total > UINT32_MAX ? 0 : static_cast<uint32_t>(total);
}

static int tensor_rank_and_dimensions(
    const Qnn_Tensor_t& tensor,
    uint32_t* dims_out,
    int max_dims
) {
    if (!dims_out || max_dims < 0) return -1;

    uint32_t rank;
    const uint32_t* dimensions;
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        rank = tensor.v1.rank;
        dimensions = tensor.v1.dimensions;
    } else if (tensor.version == QNN_TENSOR_VERSION_2) {
        rank = tensor.v2.rank;
        dimensions = tensor.v2.dimensions;
    } else {
        return -1;
    }

    if (!dimensions || rank == 0 || rank > static_cast<uint32_t>(max_dims)) return -1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (dimensions[i] == 0) return -1;
    }
    memcpy(dims_out, dimensions, rank * sizeof(uint32_t));
    return static_cast<int>(rank);
}

static bool set_client_buffer(Qnn_Tensor_t& tensor, void* data, uint32_t size) {
    if (tensor.version == QNN_TENSOR_VERSION_1) {
        tensor.v1.clientBuf.data = data;
        tensor.v1.clientBuf.dataSize = size;
        return true;
    }
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        tensor.v2.clientBuf.data = data;
        tensor.v2.clientBuf.dataSize = size;
        return true;
    }
    return false;
}

static Qnn_ErrorHandle_t load_qnn_functions(QnnContextHandle* ctx, const char* lib_path, const char* system_lib_path) {
    ctx->backend_lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!ctx->backend_lib_handle) {
        std::cerr << "[QNN] dlopen failed for backend: " << dlerror() << std::endl;
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    auto get_providers = (QnnInterfaceGetProvidersFn_t)dlsym(ctx->backend_lib_handle, "QnnInterface_getProviders");
    if (!get_providers) {
        std::cerr << "[QNN] Failed to load QnnInterface_getProviders" << std::endl;
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    const QnnInterface_t** providerList = nullptr;
    uint32_t numProviders = 0;
    if (get_providers(&providerList, &numProviders) != QNN_SUCCESS || numProviders == 0) {
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    const QnnInterface_t* qnnInterface = providerList[0];
    ctx->QnnBackend_create = qnnInterface->QNN_INTERFACE_VER_NAME.backendCreate;
    ctx->QnnBackend_free = qnnInterface->QNN_INTERFACE_VER_NAME.backendFree;
    ctx->QnnContext_createFromBinary = qnnInterface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary;
    ctx->QnnContext_free = qnnInterface->QNN_INTERFACE_VER_NAME.contextFree;
    ctx->QnnGraph_retrieve = qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve;
    ctx->QnnGraph_execute = qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute;
    if (!ctx->QnnBackend_create || !ctx->QnnBackend_free ||
        !ctx->QnnContext_createFromBinary || !ctx->QnnContext_free ||
        !ctx->QnnGraph_retrieve || !ctx->QnnGraph_execute) {
        set_last_error("[QNN] Required backend function pointer is null");
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    // Load System Library
    ctx->system_lib_handle = dlopen(system_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!ctx->system_lib_handle) {
        std::cerr << "[QNN] dlopen failed for system lib: " << dlerror() << std::endl;
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    auto get_sys_providers = (QnnSystemInterfaceGetProvidersFn_t)dlsym(ctx->system_lib_handle, "QnnSystemInterface_getProviders");
    if (!get_sys_providers) {
        std::cerr << "[QNN] Failed to load QnnSystemInterface_getProviders" << std::endl;
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    const QnnSystemInterface_t** sysProviderList = nullptr;
    uint32_t numSysProviders = 0;
    if (get_sys_providers(&sysProviderList, &numSysProviders) != QNN_SUCCESS || numSysProviders == 0) {
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    const QnnSystemInterface_t* sysInterface = sysProviderList[0];
    ctx->QnnSystemContext_create = sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate;
    ctx->QnnSystemContext_getMetadata = sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetMetaData;
    ctx->QnnSystemContext_free = sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree;
    if (!ctx->QnnSystemContext_create || !ctx->QnnSystemContext_getMetadata ||
        !ctx->QnnSystemContext_free) {
        set_last_error("[QNN] Required system function pointer is null");
        return QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED;
    }

    return QNN_SUCCESS;
}

void* qnn_backend_init() {
    QnnContextHandle* ctx = new QnnContextHandle();
    memset(ctx, 0, sizeof(QnnContextHandle));

    if (load_qnn_functions(ctx, "libQnnHtp.so", "libQnnSystem.so") != QNN_SUCCESS) {
        std::cerr << "[QNN] Failed to load QNN libraries" << std::endl;
        if (g_last_error.empty()) set_last_error("[QNN] Failed to load QNN libraries");
        if (ctx->system_lib_handle) dlclose(ctx->system_lib_handle);
        if (ctx->backend_lib_handle) dlclose(ctx->backend_lib_handle);
        delete ctx;
        return nullptr;
    }

    if (ctx->QnnBackend_create(nullptr, nullptr, &ctx->backend) != QNN_SUCCESS) {
        std::cerr << "[QNN] Backend create failed" << std::endl;
        set_last_error("[QNN] Backend create failed");
        dlclose(ctx->system_lib_handle);
        dlclose(ctx->backend_lib_handle);
        delete ctx;
        return nullptr;
    }

    return ctx;
}

// Releases everything qnn_load_context() attaches to a handle (context,
// graph, tensor metadata, retained system context) without touching the
// backend/library handles. Called both from qnn_backend_destroy() and, if a
// handle is ever reloaded in place, from qnn_load_context() itself — a
// caller is not required to allocate a fresh handle per load, so a second
// load must not leak the first one's context/tensors/system context.
static void free_loaded_context(QnnContextHandle* ctx) {
    if (ctx->context && ctx->QnnContext_free) {
        ctx->QnnContext_free(ctx->context, nullptr);
    }
    ctx->context = nullptr;
    ctx->graph = nullptr;

    if (ctx->inputs) {
        for (uint32_t i = 0; i < ctx->num_inputs; ++i) {
            if (ctx->inputs[i].version == QNN_TENSOR_VERSION_1) delete[] ctx->inputs[i].v1.dimensions;
            else if (ctx->inputs[i].version == QNN_TENSOR_VERSION_2) delete[] ctx->inputs[i].v2.dimensions;
        }
        delete[] ctx->inputs;
        ctx->inputs = nullptr;
    }
    ctx->num_inputs = 0;

    if (ctx->outputs) {
        for (uint32_t i = 0; i < ctx->num_outputs; ++i) {
            if (ctx->outputs[i].version == QNN_TENSOR_VERSION_1) delete[] ctx->outputs[i].v1.dimensions;
            else if (ctx->outputs[i].version == QNN_TENSOR_VERSION_2) delete[] ctx->outputs[i].v2.dimensions;
        }
        delete[] ctx->outputs;
        ctx->outputs = nullptr;
    }
    ctx->num_outputs = 0;

    if (ctx->system_context && ctx->QnnSystemContext_free) {
        ctx->QnnSystemContext_free(ctx->system_context);
    }
    ctx->system_context = nullptr;
}

bool qnn_load_context(void* ctx_ptr, const char* model_path) {
    if (!ctx_ptr || !model_path) {
        set_last_error("[QNN] Null context or model path");
        return false;
    }
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);

    // A handle previously loaded via this function (or destroyed and
    // recreated) must not leak its old context/tensors/system context when
    // reloaded in place.
    free_loaded_context(ctx);

    FILE* file = fopen(model_path, "rb");
    if (!file) {
        std::cerr << "[QNN] Failed to open context binary: " << model_path << std::endl;
        set_last_error("[QNN] Failed to open context binary");
        return false;
    }

    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<uint8_t> buffer(fileSize);
    if (fread(buffer.data(), 1, fileSize, file) != fileSize) {
        fclose(file);
        set_last_error("[QNN] Failed to read complete context binary");
        return false;
    }
    fclose(file);

    // Get metadata using System API
    QnnSystemContext_Handle_t sysCtx = nullptr;
    if (ctx->QnnSystemContext_create(&sysCtx) != QNN_SUCCESS) {
        std::cerr << "[QNN] Failed to create system context" << std::endl;
        set_last_error("[QNN] Failed to create system context");
        return false;
    }

    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    if (ctx->QnnSystemContext_getMetadata(sysCtx, buffer.data(), buffer.size(), &binaryInfo) != QNN_SUCCESS) {
        std::cerr << "[QNN] Failed to get metadata from context binary" << std::endl;
        set_last_error("[QNN] Failed to get metadata from context binary");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    uint32_t numGraphs = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;

    if (!binaryInfo) {
        std::cerr << "[QNN] No binary info returned" << std::endl;
        set_last_error("[QNN] No binary info returned");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
        graphs = binaryInfo->contextBinaryInfoV1.graphs;
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
        graphs = binaryInfo->contextBinaryInfoV2.graphs;
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
        graphs = binaryInfo->contextBinaryInfoV3.graphs;
    } else {
        std::cerr << "[QNN] Unsupported binary info version: " << binaryInfo->version << std::endl;
        set_last_error("[QNN] Unsupported binary info version");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    if (numGraphs == 0 || !graphs) {
        std::cerr << "[QNN] No graphs found in context binary" << std::endl;
        set_last_error("[QNN] No graphs found in context binary");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    // Assume 1 graph for simplicity
    auto graphInfo = graphs[0];
    const char* graphName = nullptr;
    uint32_t numGraphInputs = 0;
    Qnn_Tensor_t* graphInputs = nullptr;
    uint32_t numGraphOutputs = 0;
    Qnn_Tensor_t* graphOutputs = nullptr;

    if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        graphName = graphInfo.graphInfoV1.graphName;
        numGraphInputs = graphInfo.graphInfoV1.numGraphInputs;
        graphInputs = graphInfo.graphInfoV1.graphInputs;
        numGraphOutputs = graphInfo.graphInfoV1.numGraphOutputs;
        graphOutputs = graphInfo.graphInfoV1.graphOutputs;
    } else if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        graphName = graphInfo.graphInfoV2.graphName;
        numGraphInputs = graphInfo.graphInfoV2.numGraphInputs;
        graphInputs = graphInfo.graphInfoV2.graphInputs;
        numGraphOutputs = graphInfo.graphInfoV2.numGraphOutputs;
        graphOutputs = graphInfo.graphInfoV2.graphOutputs;
    } else if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        graphName = graphInfo.graphInfoV3.graphName;
        numGraphInputs = graphInfo.graphInfoV3.numGraphInputs;
        graphInputs = graphInfo.graphInfoV3.graphInputs;
        numGraphOutputs = graphInfo.graphInfoV3.numGraphOutputs;
        graphOutputs = graphInfo.graphInfoV3.graphOutputs;
    } else {
        std::cerr << "[QNN] Unsupported graph info version: " << graphInfo.version << std::endl;
        set_last_error("[QNN] Unsupported graph info version");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    if (!graphName || !graphInputs || !graphOutputs) {
        std::cerr << "[QNN] Invalid graph metadata: null graph name or tensor array" << std::endl;
        set_last_error("[QNN] Invalid graph metadata");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    if (numGraphInputs != 1 || numGraphOutputs != 1) {
        std::cerr << "[QNN] Exactly one input and one output are currently supported" << std::endl;
        set_last_error("[QNN] Exactly one input and one output are currently supported");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    if (tensor_precision(graphInputs[0]) < 0 || tensor_precision(graphOutputs[0]) < 0 ||
        tensor_expected_bytes(graphInputs[0]) == 0 || tensor_expected_bytes(graphOutputs[0]) == 0) {
        std::cerr << "[QNN] Unsupported or malformed input/output tensor metadata" << std::endl;
        set_last_error("[QNN] Unsupported or malformed input/output tensor metadata");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    Qnn_ProfileHandle_t profile = nullptr;
    if (ctx->QnnContext_createFromBinary(
        ctx->backend,
        nullptr, // device
        nullptr, // custom config
        buffer.data(),
        buffer.size(),
        &ctx->context,
        profile
    ) != QNN_SUCCESS) {
        std::cerr << "[QNN] Context creation failed" << std::endl;
        set_last_error("[QNN] Context creation failed");
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    // Retrieve graph handle
    if (ctx->QnnGraph_retrieve(ctx->context, graphName, &ctx->graph) != QNN_SUCCESS) {
        std::cerr << "[QNN] Failed to retrieve graph: " << graphName << std::endl;
        set_last_error("[QNN] Failed to retrieve graph");
        if (ctx->QnnContext_free) {
            ctx->QnnContext_free(ctx->context, nullptr);
            ctx->context = nullptr;
        }
        ctx->QnnSystemContext_free(sysCtx);
        return false;
    }

    // Save tensor metadata. sysCtx is retained in ctx->system_context and
    // freed once in qnn_backend_destroy(), not here — but graphInputs/
    // graphOutputs are separate heap blocks owned by the QNN system-context
    // library that this wrapper does not otherwise keep alive, so the
    // dimension arrays are still deep-copied defensively rather than aliased.
    ctx->num_inputs = numGraphInputs;
    ctx->inputs = new Qnn_Tensor_t[ctx->num_inputs];
    for (uint32_t i = 0; i < ctx->num_inputs; ++i) {
        ctx->inputs[i] = graphInputs[i];
        if (ctx->inputs[i].version == QNN_TENSOR_VERSION_1) {
            uint32_t rank = ctx->inputs[i].v1.rank;
            ctx->inputs[i].v1.dimensions = new uint32_t[rank];
            memcpy(ctx->inputs[i].v1.dimensions, graphInputs[i].v1.dimensions, rank * sizeof(uint32_t));
        } else if (ctx->inputs[i].version == QNN_TENSOR_VERSION_2) {
            uint32_t rank = ctx->inputs[i].v2.rank;
            ctx->inputs[i].v2.dimensions = new uint32_t[rank];
            memcpy(ctx->inputs[i].v2.dimensions, graphInputs[i].v2.dimensions, rank * sizeof(uint32_t));
        }
    }

    ctx->num_outputs = numGraphOutputs;
    ctx->outputs = new Qnn_Tensor_t[ctx->num_outputs];
    for (uint32_t i = 0; i < ctx->num_outputs; ++i) {
        ctx->outputs[i] = graphOutputs[i];
        if (ctx->outputs[i].version == QNN_TENSOR_VERSION_1) {
            uint32_t rank = ctx->outputs[i].v1.rank;
            ctx->outputs[i].v1.dimensions = new uint32_t[rank];
            memcpy(ctx->outputs[i].v1.dimensions, graphOutputs[i].v1.dimensions, rank * sizeof(uint32_t));
        } else if (ctx->outputs[i].version == QNN_TENSOR_VERSION_2) {
            uint32_t rank = ctx->outputs[i].v2.rank;
            ctx->outputs[i].v2.dimensions = new uint32_t[rank];
            memcpy(ctx->outputs[i].v2.dimensions, graphOutputs[i].v2.dimensions, rank * sizeof(uint32_t));
        }
    }

    ctx->system_context = sysCtx;
    return true;
}

bool qnn_execute_graph(void* ctx_ptr, const void* input_data, int input_bytes, void* output_data, int output_bytes) {
    if (!ctx_ptr || !input_data || !output_data || input_bytes <= 0 || output_bytes <= 0) {
        set_last_error("[QNN] Invalid execute arguments");
        return false;
    }
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);

    if (!ctx->graph || !ctx->QnnGraph_execute) {
        set_last_error("[QNN] Graph is not loaded");
        return false;
    }

    if (ctx->num_inputs != 1 || ctx->num_outputs != 1 || !ctx->inputs || !ctx->outputs) {
        set_last_error("[QNN] Invalid loaded tensor metadata");
        return false;
    }

    // Validate the caller's buffer sizes against the real context tensors before
    // handing raw pointers to QnnGraph_execute. A too-small output buffer would
    // let the graph write out of bounds; a too-large one silently leaves a
    // stale/garbage tail (both reported as issue #54). A width of 0 means the
    // tensor's dtype/dims aren't introspectable here, so the check is skipped.
    uint32_t expected_in = tensor_expected_bytes(ctx->inputs[0]);
    if (expected_in == 0 || expected_in != static_cast<uint32_t>(input_bytes)) {
        std::cerr << "[QNN] Input buffer size mismatch: caller gave " << input_bytes
                  << " bytes, context tensor expects " << expected_in << std::endl;
        set_last_error("[QNN] Input buffer size mismatch");
        return false;
    }
    uint32_t expected_out = tensor_expected_bytes(ctx->outputs[0]);
    if (expected_out == 0 || expected_out != static_cast<uint32_t>(output_bytes)) {
        std::cerr << "[QNN] Output buffer size mismatch: caller gave " << output_bytes
                  << " bytes, context tensor expects " << expected_out
                  << " (check the engine's reported output precision)" << std::endl;
        set_last_error("[QNN] Output buffer size mismatch");
        return false;
    }

    if (!set_client_buffer(ctx->inputs[0], const_cast<void*>(input_data), static_cast<uint32_t>(input_bytes)) ||
        !set_client_buffer(ctx->outputs[0], output_data, static_cast<uint32_t>(output_bytes))) {
        set_last_error("[QNN] Failed to bind client buffers");
        return false;
    }

    // Execute the graph synchronously
    Qnn_ErrorHandle_t status = ctx->QnnGraph_execute(
        ctx->graph,
        ctx->inputs,
        ctx->num_inputs,
        ctx->outputs,
        ctx->num_outputs,
        nullptr, // profile handle
        nullptr  // signal handle
    );

    if (status != QNN_SUCCESS) {
        set_last_error("[QNN] Graph execution failed");
        return false;
    }
    return true;
}

void qnn_backend_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);

    free_loaded_context(ctx);

    if (ctx->backend && ctx->QnnBackend_free) {
        ctx->QnnBackend_free(ctx->backend);
    }

    if (ctx->system_lib_handle) {
        dlclose(ctx->system_lib_handle);
    }

    if (ctx->backend_lib_handle) {
        dlclose(ctx->backend_lib_handle);
    }

    delete ctx;
}

} // extern "C"

extern "C" {
int qnn_get_num_inputs(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    return ctx->num_inputs;
}

int qnn_get_num_outputs(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    return ctx->num_outputs;
}

int qnn_get_input_dims(void* ctx_ptr, int tensor_idx, uint32_t* dims_out, int max_dims) {
    if (!ctx_ptr) return -1;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_inputs) return -1;

    return tensor_rank_and_dimensions(ctx->inputs[tensor_idx], dims_out, max_dims);
}

int qnn_get_output_dims(void* ctx_ptr, int tensor_idx, uint32_t* dims_out, int max_dims) {
    if (!ctx_ptr) return -1;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_outputs) return -1;

    return tensor_rank_and_dimensions(ctx->outputs[tensor_idx], dims_out, max_dims);
}

const char* qnn_get_input_name(void* ctx_ptr, int tensor_idx) {
    if (!ctx_ptr) return nullptr;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_inputs) return nullptr;
    return tensor_name(ctx->inputs[tensor_idx]);
}

const char* qnn_get_output_name(void* ctx_ptr, int tensor_idx) {
    if (!ctx_ptr) return nullptr;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_outputs) return nullptr;
    return tensor_name(ctx->outputs[tensor_idx]);
}

int qnn_get_input_precision(void* ctx_ptr, int tensor_idx) {
    if (!ctx_ptr) return -1;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_inputs) return -1;
    return tensor_precision(ctx->inputs[tensor_idx]);
}

int qnn_get_output_precision(void* ctx_ptr, int tensor_idx) {
    if (!ctx_ptr) return -1;
    QnnContextHandle* ctx = static_cast<QnnContextHandle*>(ctx_ptr);
    if (tensor_idx < 0 || static_cast<uint32_t>(tensor_idx) >= ctx->num_outputs) return -1;
    return tensor_precision(ctx->outputs[tensor_idx]);
}

const char* qnn_last_error_message() {
    return g_last_error.c_str();
}
}
