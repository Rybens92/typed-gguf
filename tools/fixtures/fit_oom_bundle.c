/* A fake llama.cpp bundle that fails the way ggml_vulkan did on the operator's box.
 *
 * Vehicle for card t_8cb0a05e requirement 5: a *fake allocation failure* that needs no GPU, so the
 * ladder / classification / error code can be exercised anywhere (sandbox rehearsal, CI, and the
 * host gate on a box whose desktop is not busy).
 *
 *   cc -shared -fPIC -o libllama.so fixtures/fit_oom_bundle.c        (libllama.so)
 *   cc -shared -fPIC -o libggml.so  fixtures/fit_oom_bundle.c        (libggml.so)
 *
 * Behaviour:
 *   - `llama_model_load_from_file(path, params)` prints the operator's exact lines through the
 *     installed log callback and returns NULL while `params.n_gpu_layers > 0` (the descriptor of
 *     "nothing fits in the device"), and returns a non-NULL fake model when every layer is on the
 *     CPU — which is precisely the ladder's last rung.
 *   - the required symbols `ctypes_binding._bind` insists on exist (the ones this probe never
 *     calls are stubs): the point is that production's real dlopen path runs, not a mock.
 *   - `llama_model_spark2_5` exists so the arch pre-flight (`capability.require_arch`) passes for
 *     the pinned model's architecture.
 *   - `ggml_backend_dev_by_name("CPU")` answers a non-NULL handle, because a `cpu` row resolves
 *     the device it is allowed to compute on BEFORE the load ladder (card t_55de5779's pin in
 *     `session.open_model`). Without it the fake bundle is refused with `E_RUNTIME_SYMBOLS` and
 *     the OOM path this fixture exists for is never reached — which is exactly what the first
 *     live matrix run proved (card t_8dab8b3a: job 108153215436, step "the placement retry
 *     answers a typed row, never E_INTERNAL").
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* `struct llama_model_params` — field order/sizes verbatim from include/llama.h @ b11026 and
 * mirrored by ctypes_binding.llama_model_params (the ctypes caller reads n_gpu_layers at the same
 * offset, so a drift here would show up as the fake never offloading). */
struct llama_model_params {
    void *devices;
    void *tensor_buft_overrides;
    int n_gpu_layers;
    int split_mode;
    int load_mode;
    int lazy_mode;
    int main_gpu;
    float *tensor_split;
    void *progress_callback;
    void *progress_callback_user_data;
    void *kv_overrides;
    _Bool vocab_only;
    _Bool check_tensors;
    _Bool use_extra_bufts;
    _Bool no_host;
    _Bool no_alloc;
    _Bool load_mtp;
};

typedef void (*ggml_log_callback)(int level, const char *text, void *user_data);

static ggml_log_callback g_log = NULL;
static void *g_log_user_data = NULL;
static int g_fake_model = 0;          /* the non-NULL pointer handed back for a CPU load */
static int g_fake_vocab = 0;

void llama_log_set(ggml_log_callback callback, void *user_data) {
    g_log = callback;
    g_log_user_data = user_data;
}

void *llama_log_get(void) {
    return (void *) g_log;
}

static void emit(const char *text) {
    if (g_log) {
        g_log(4 /* GGML_LOG_LEVEL_ERROR */, text, g_log_user_data);
    }
    fputs(text, stderr);
}

struct llama_model_params llama_model_default_params(void) {
    struct llama_model_params params;
    memset(&params, 0, sizeof(params));
    params.split_mode = 1;            /* LLAMA_SPLIT_MODE_LAYER */
    return params;
}

void *llama_model_load_from_file(const char *path, struct llama_model_params params) {
    (void) path;
    /* `TYPED_GGUF_FAKE_OOM_ALL=1` fails even the CPU-only rung: the "nothing fits" world that must
     * answer E_BACKEND_OOM instead of a load error. */
    if (params.n_gpu_layers > 0 || getenv("TYPED_GGUF_FAKE_OOM_ALL") != NULL) {
        emit("ggml_vulkan: Device memory allocation of size 1058982400 failed.\n");
        emit("ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory\n");
        emit("alloc_tensor_range: failed to allocate Vulkan0 buffer of size 1058982400\n");
        emit("llama_model_load: error loading model: unable to allocate Vulkan0 buffer\n");
        return NULL;
    }
    return &g_fake_model;
}

void *llama_model_get_vocab(void *model) { (void) model; return &g_fake_vocab; }
int llama_vocab_n_tokens(void *vocab) { (void) vocab; return 1024; }
int llama_model_n_layer(void *model) { (void) model; return 32; }
int llama_model_n_embd(void *model) { (void) model; return 2560; }
void llama_model_free(void *model) { (void) model; }
void llama_backend_init(void) {}
void llama_backend_free(void) {}

/* The backend loader: called with the bundle directory before any model load (PoC pitfall 1). */
void ggml_backend_load_all(void) {}
void ggml_backend_load_all_from_path(const char *dir) { (void) dir; }

/* The device query a *CPU-pinned* load resolves before the ladder: a `cpu` row must be able to
 * name the device it may compute on (`ctypes_binding.cpu_device` -> `ggml_backend_dev_by_name`).
 * The handle is only ever tested for non-NULL and passed back inside `llama_model_params.devices`,
 * so a real address inside this library is the honest stand-in — and returning NULL here would
 * make production refuse the bundle (E_RUNTIME_SYMBOLS) before the OOM ladder runs.
 *
 * `-DTYPED_GGUF_FIXTURE_NO_CPU_DEVICE` builds the fixture as it was BEFORE card t_8dab8b3a: the
 * suite's RED control (`tests/test_fit_oom_fixture.py`), so the refusal this gate produces stays
 * reproducible instead of being a story about a run nobody can replay. */
#ifndef TYPED_GGUF_FIXTURE_NO_CPU_DEVICE
static int g_cpu_device = 0;
void *ggml_backend_dev_by_name(const char *name) {
    if (name != NULL && strcmp(name, "CPU") == 0) {
        return (void *) &g_cpu_device;
    }
    return NULL;
}
#endif

/* The arch the pinned model needs (the real bundle exports this symbol; the pre-flight scans the
 * file's bytes for the name). */
void llama_model_spark2_5(void) {}

/* ------------------------------------------------------------------ required stubs
 * `ctypes_binding._bind` refuses a bundle that lacks any of these. None of them is reached by the
 * OOM probe, so plain stubs are enough — but they must EXIST, or production answers
 * E_RUNTIME_SYMBOLS before the interesting code runs. */
void llama_free(void) {}
void llama_synchronize(void) {}
void llama_memory_seq_cp(void) {}
void llama_memory_seq_keep(void) {}
void llama_memory_seq_rm(void) {}
void llama_state_seq_get_size(void) {}
void llama_state_seq_save_file(void) {}
void llama_state_seq_load_file(void) {}
void llama_get_memory(void) {}
void llama_decode(void) {}
void llama_get_logits_ith(void) {}
void llama_tokenize(void) {}
void llama_token_to_piece(void) {}
void llama_chat_apply_template(void) {}
void llama_chat_builtin_templates(void) {}
void llama_model_meta_val_str(void) {}
void llama_model_chat_template(void) {}
void llama_batch_init(void) {}
void llama_batch_free(void) {}
void llama_batch_get_one(void) {}
void llama_n_ctx(void) {}
void llama_n_seq_max(void) {}
void *llama_context_default_params(void) { return NULL; }
void *llama_init_from_model(void) { return NULL; }
