# Getting Started

This package is designed for continuous wakeword scoring in Go applications.

> Courtesy: [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)

## 1) Install

```bash
go get github.com/rajeshpachaikani/openWakeWord-go
```

## 2) Create an engine

```go
engine := openwakeword.NewEngine(openwakeword.Options{})
```

Defaults:

- Sample rate: `16000`
- Frame size: `1280`

## 3) Load model files

```go
_, _ = engine.AddModel("/absolute/path/to/alexa.onnx")
_, _ = engine.AddModel("/absolute/path/to/hey_jarvis.tflite")
```

Supported model extensions:

- `.onnx`
- `.tflite`

## 4) Stream audio frames

Provide signed 16-bit mono PCM audio (`[]int16`), sampled at 16 kHz.

```go
scores, err := engine.Predict(frame)
```

`Predict` returns a `map[string]float32` with one score per loaded model.

## 5) Trigger on threshold

A common integration pattern:

```go
if score >= 0.55 {
    // wakeword detected
}
```

Tune thresholds per model and deployment environment.
