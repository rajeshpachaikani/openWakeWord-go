# Examples

## Detect from microphone (Go)

A native Go microphone streaming example is included at:

`examples/go/detect_from_microphone`

Run it with one or more model paths:

```bash
cd examples/go/detect_from_microphone
go run . -models /absolute/path/to/model.onnx
```

Multiple models:

```bash
go run . -models /m/alexa.onnx,/m/hey_jarvis.tflite
```

Useful flags:

- `-threshold` (default `0.55`)
- `-cooldown` (default `1.5s`)
