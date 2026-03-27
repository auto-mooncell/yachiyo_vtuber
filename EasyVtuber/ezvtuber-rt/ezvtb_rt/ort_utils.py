import onnxruntime as ort


def get_ort_runtime(device_id: int = 0):
    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        return {
            "provider": "DmlExecutionProvider",
            "device": "dml",
            "providers": [("DmlExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"],
        }
    if "CUDAExecutionProvider" in available:
        return {
            "provider": "CUDAExecutionProvider",
            "device": "cuda",
            "providers": [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": device_id,
                        "execution_mode": "parallel",
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ],
        }
    return {
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
    }


def createORTSession(model_path: str, device_id: int = 0):
    runtime = get_ort_runtime(device_id)
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=runtime["providers"],
    )
    return session
