# openWakeWord-go

A Go package for stream-oriented wakeword detection using openWakeWord model files (`.onnx` and `.tflite`).

> Courtesy: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)

This repository is focused on Go usage so it is easy to consume from:

- `go get github.com/rajeshpachaikani/openWakeWord-go`
- [pkg.go.dev](https://pkg.go.dev/github.com/rajeshpachaikani/openWakeWord-go)

## Install

```bash
go get github.com/rajeshpachaikani/openWakeWord-go
```

## Quick start

```go
package main

import (
"log"
"os"

openwakeword "github.com/rajeshpachaikani/openWakeWord-go"
)

func main() {
engine := openwakeword.NewEngine(openwakeword.Options{})

if _, err := engine.AddModel("/absolute/path/to/alexa.tflite"); err != nil {
log.Fatal(err)
}

frame := make([]int16, 1280) // 80 ms @ 16 kHz mono PCM
scores, err := engine.Predict(frame)
if err != nil {
log.Fatal(err)
}

for model, score := range scores {
log.Printf("model=%s score=%.4f", model, score)
}

_ = os.Stdout
}
```

## How it works

- Create an `Engine` with `NewEngine`.
- Load one or more model files with `AddModel`.
- Continuously pass audio frames to `Predict`.
- Read per-model scores (`0.0` to `1.0`) from the returned map.

## Audio requirements

- PCM format: signed 16-bit (`[]int16`)
- Sample rate: `16000` Hz
- Recommended frame size: `1280` samples (80 ms)

If your audio callback gives smaller chunks, call `Predict` repeatedly. The engine buffers partial frames internally.

## Example app

A microphone streaming example is provided:

```bash
cd examples/go/detect_from_microphone
go run . -models /absolute/path/to/model.onnx
```

Pass multiple models with a comma-separated list:

```bash
go run . -models /m/alexa.onnx,/m/hey_jarvis.tflite
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [API usage and behavior](docs/api_usage.md)
- [Examples](examples/README.md)

## License

Apache-2.0
