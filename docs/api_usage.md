# API Usage and Behavior

## `NewEngine(opts Options) *Engine`

Creates a detection engine.

- `SampleRate`: defaults to `16000` when zero
- `FrameSize`: defaults to `1280` when zero

## `(*Engine) AddModel(path string) (string, error)`

Loads a model from disk and registers it by base filename.

Example:

- `/models/alexa.onnx` -> model key `"alexa"`

Errors if:

- engine is nil
- file cannot be read
- extension is not `.onnx` or `.tflite`

## `(*Engine) Predict(frame []int16) (map[string]float32, error)`

Processes input audio and returns current model scores.

Behavior details:

- Requires at least one model loaded
- Buffers partial frames internally until `FrameSize` is reached
- Returns latest known score per model when no full frame is available
- Score range is `0.0` to `1.0`

## `(*Engine) Reset()`

Clears internal frame remainders and model score history.

## Integration notes

- For best latency/efficiency, use 80 ms chunks (`1280` samples at 16 kHz)
- Apply model-specific thresholds and cooldown logic in your application layer
- Use the provided example under `examples/go/detect_from_microphone` as a starting point
