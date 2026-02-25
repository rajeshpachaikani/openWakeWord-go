// Package openwakeword provides a Go wakeword detection engine for stream-oriented
// scoring with openWakeWord model files (.onnx and .tflite).
//
// Courtesy: dscripka/openWakeWord.
//
// Typical usage:
//
//engine := openwakeword.NewEngine(openwakeword.Options{})
//_, _ = engine.AddModel("/absolute/path/to/alexa.onnx")
//scores, _ := engine.Predict(frame) // frame is []int16 PCM audio
//
// The engine is intended for real-time applications where audio arrives in chunks.
// It buffers partial input and emits one score per loaded model.
package openwakeword
