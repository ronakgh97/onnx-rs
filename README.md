Minimal **onnx runtime** for running `fp32 precision` models on **x86_64** CPUs ⚡

Notes:

- Runtime is very constrained, only few type of models and certain runtime behaviors are supported (thanks to onnx spec
  for making this harder)
- Model parser is quite weak and complex to implement and needs to have a reasonably simple graph
- Nothing is assumed, anything "alien" to this runtime's scope is rejected and panics instead of silently ignoring or
  assuming, which may lead to bad undesired outputs
- Operators correctness tests are not covered yet, so expect some undesired behaviors/outputs on sophisticated operators
  used by the model

> The goal is that:
> "`If model passes the checks during loading, it should run without any silent errors or corrupted outputs`"

Benchmarks and correctness:

[ESRGAN](https://aihub.qualcomm.com/compute/models/esrgan) 69.80 MB

| Runtime | Time  | Input                                               | Output                                                        |
|---------|-------|-----------------------------------------------------|---------------------------------------------------------------|
| ONNX_RS | 4.61s | <img src="images/samples/sample_1.png" width="512"> | <img src="images/upscaled/esrgan_onnx_rs_x4.png" width="512"> |
| ORT     | 1.16s | <img src="images/samples/sample_1.png" width="512"> | <img src="images/upscaled/esrgan_ort_x4.png" width="512">     |

[Real-ERSRGAN-Anime](https://huggingface.co/deepghs/imgutils-models/blob/main/real_esrgan/RealESRGAN_x4plus_anime_6B.onnx)
17.21 MB

| Runtime | Time     | Input                                               | Output                                                                   |
|---------|----------|-----------------------------------------------------|--------------------------------------------------------------------------|
| ONNX_RS | 1.40s    | <img src="images/samples/sample_2.png" width="512"> | <img src="images/upscaled/real_esrgan_anime_onnx_rs_x4.png" width="512"> |
| ORT     | 550.08ms | <img src="images/samples/sample_2.png" width="512"> | <img src="images/upscaled/real_esrgan_anime_ort_x4.png" width="512">     |

[ClearReality-SPAN](https://huggingface.co/huggingworld/onnx-image-models/blob/main/4x-ClearRealityV1.onnx) 1.72 MB

| Runtime | Time     | Input                                               | Output                                                               |
|---------|----------|-----------------------------------------------------|----------------------------------------------------------------------|
| ONNX_RS | 220.54ms | <img src="images/samples/sample_3.png" width="512"> | <img src="images/upscaled/clear_reality_onnx_rs_x4.png" width="512"> |
| ORT     | 71.60ms  | <img src="images/samples/sample_3.png" width="512"> | <img src="images/upscaled/clear_reality_ort_x4.png" width="512">     |

[SRGAN](https://aihub.qualcomm.com/compute/models/srgan) 6.92 MB

| Runtime | Time     | Input                                               | Output                                                       |
|---------|----------|-----------------------------------------------------|--------------------------------------------------------------|
| ONNX_RS | 1.14s    | <img src="images/samples/sample_4.png" width="512"> | <img src="images/upscaled/srgan_onnx_rs_x4.png" width="512"> |
| ORT     | 335.56ms | <img src="images/samples/sample_4.png" width="512"> | <img src="images/upscaled/srgan_ort_x4.png" width="512">     |

[Upscayl-Standard](https://huggingface.co/huggingworld/onnx-image-models/blob/main/upscayl-standard-4x.onnx) 67.16 MB

| Runtime | Time  | Input                                               | Output                                                                  |
|---------|-------|-----------------------------------------------------|-------------------------------------------------------------------------|
| ONNX_RS | 4.55s | <img src="images/samples/sample_5.png" width="512"> | <img src="images/upscaled/upscayl_standard_onnx_rs_x4.png" width="512"> |
| ORT     | 1.56s | <img src="images/samples/sample_5.png" width="512"> | <img src="images/upscaled/upscayl_standard_ort_x4.png" width="512">     |

[Upscayl-High-Fidelity](https://huggingface.co/huggingworld/onnx-image-models/blob/main/upscayl-high-fidelity-4x.onnx)
67.16 MB

| Runtime | Time   | Input                                               | Output                                                                       |
|---------|--------|-----------------------------------------------------|------------------------------------------------------------------------------|
| ONNX_RS | 4.54ms | <img src="images/samples/sample_6.png" width="512"> | <img src="images/upscaled/upscayl_high_fidelity_onnx_rs_x4.png" width="512"> |
| ORT     | 1.22ms | <img src="images/samples/sample_6.png" width="512"> | <img src="images/upscaled/upscayl_high_fidelity_ort_x4.png" width="512">     |

---

More **models/kernels** will be supported in the future...