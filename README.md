Minimal **onnx runtime** for running `f32 precision` models on **x86_64** CPUs ⚡

Notes:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making this harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph
- Nothing is assumed, anything "alien" to this runtime's scope is rejected and panics instead of silently ignoring or
  assuming, which may lead to bad undesired outputs
- Operators correctness tests are covered yet, so expect some undesired behaviors/outputs on sophisticated operators
  used by the model

> The goal is that:
> "`If model passes the checks during loading, it should run without any silent errors or corrupted outputs`"

Benchmarks and correctness:

Running [Esrgan](https://aihub.qualcomm.com/compute/models/esrgan) 64.02 MB

| Runtime | Time   | Input                                       | Output                                               |
|---------|--------|---------------------------------------------|------------------------------------------------------|
| ONNX_RS | 5.27s  | <img src="images/sample_1.png" width="512"> | <img src="images/esrgan_onnx_rs_x4.png" width="512"> |
| ORT     | 1.20ms | <img src="images/sample_1.png" width="512"> | <img src="images/esrgan_ort_x4.png" width="512">     |

Running [Srgan](https://aihub.qualcomm.com/compute/models/srgan) 5.92 MB

| Runtime | Time     | Input                                       | Output                                              |
|---------|----------|---------------------------------------------|-----------------------------------------------------|
| ONNX_RS | 1.20s   | <img src="images/sample_2.png" width="512"> | <img src="images/srgan_onnx_rs_x4.png" width="512"> |
| ORT     | 230.77ms | <img src="images/sample_2.png" width="512"> | <img src="images/srgan_ort_x4.png" width="512">     |

Running [LSDIR Compact v2](https://openmodeldb.info/models/4x-LSDIRCompact-v2) 2.42 MB

| Runtime | Time     | Input                                       | Output                                                     |
|---------|----------|---------------------------------------------|------------------------------------------------------------|
| ONNX_RS | 211.37ms | <img src="images/sample_3.png" width="512"> | <img src="images/lsdircompact_onnx_rs_x4.png" width="512"> |
| ORT     | 71.27ms  | <img src="images/sample_3.png" width="512"> | <img src="images/lsdircompact_ort_x4.png" width="512">     |

Running [4xClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1) 1.72 MB

| Runtime | Time     | Input                                       | Output                                                     |
|---------|----------|---------------------------------------------|------------------------------------------------------------|
| ONNX_RS | 260.23ms | <img src="images/sample_4.png" width="512"> | <img src="images/clearreality_onnx_rs_x4.png" width="512"> |
| ORT     | 72.33ms  | <img src="images/sample_4.png" width="512"> | <img src="images/clearreality_ort_x4.png" width="512">     |
