import onnxruntime as ort

def createORTSession(model_path:str, device_id:int = 0):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.enable_mem_pattern = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    provider_options = [{'device_id':device_id, "execution_mode": "parallel", "arena_extend_strategy": "kSameAsRequested"}]
    session = ort.InferenceSession(model_path, sess_options=options, providers=['CUDAExecutionProvider'], provider_options=None)
    return session