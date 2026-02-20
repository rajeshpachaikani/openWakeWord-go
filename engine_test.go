package openwakeword

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAddModelSupportsONNXAndTFLite(t *testing.T) {
	tflite := writeTestModel(t, "alexa.tflite", []byte{1, 2, 3, 4})
	onnx := writeTestModel(t, "hey_jarvis.onnx", []byte{4, 3, 2, 1})

	engine := NewEngine(Options{})
	if _, err := engine.AddModel(tflite); err != nil {
		t.Fatalf("add tflite model: %v", err)
	}
	if _, err := engine.AddModel(onnx); err != nil {
		t.Fatalf("add onnx model: %v", err)
	}

	got, err := engine.Predict(make([]int16, 1280))
	if err != nil {
		t.Fatalf("predict: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 model scores, got %d", len(got))
	}
}

func TestPredictBuffersStreamedFrames(t *testing.T) {
	tflite := writeTestModel(t, "timer.tflite", []byte{9, 9, 9})

	engine := NewEngine(Options{})
	if _, err := engine.AddModel(tflite); err != nil {
		t.Fatalf("add model: %v", err)
	}
	if _, err := engine.Predict(make([]int16, 640)); err != nil {
		t.Fatalf("predict first half frame: %v", err)
	}
	got, err := engine.Predict(make([]int16, 640))
	if err != nil {
		t.Fatalf("predict second half frame: %v", err)
	}
	if _, ok := got["timer"]; !ok {
		t.Fatalf("expected timer score in output: %+v", got)
	}
}

func TestRejectUnsupportedModelExtension(t *testing.T) {
	invalid := writeTestModel(t, "bad.bin", []byte{0xAA})
	engine := NewEngine(Options{})
	if _, err := engine.AddModel(invalid); err == nil {
		t.Fatal("expected unsupported model extension error")
	}
}

func writeTestModel(t *testing.T, name string, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write test model %q: %v", path, err)
	}
	return path
}
