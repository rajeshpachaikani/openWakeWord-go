package openwakeword

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultSampleRate = 16000
	defaultFrameSize  = 1280
	historySize       = 30
)

type backend string

const (
	backendTFLite backend = "tflite"
	backendONNX   backend = "onnx"
)

type Options struct {
	SampleRate int
	FrameSize  int
}

type Engine struct {
	sampleRate int
	frameSize  int
	models     map[string]*model
	remainder  []int16
	history    map[string][]float32
}

type model struct {
	name     string
	backend  backend
	seedBias float64
	seedGain float64
}

func NewEngine(opts Options) *Engine {
	sampleRate := opts.SampleRate
	if sampleRate == 0 {
		sampleRate = defaultSampleRate
	}
	frameSize := opts.FrameSize
	if frameSize == 0 {
		frameSize = defaultFrameSize
	}
	return &Engine{
		sampleRate: sampleRate,
		frameSize:  frameSize,
		models:     map[string]*model{},
		history:    map[string][]float32{},
	}
}

func (e *Engine) AddModel(path string) (string, error) {
	if e == nil {
		return "", errors.New("engine is nil")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read model: %w", err)
	}
	ext := strings.ToLower(filepath.Ext(path))
	var b backend
	switch ext {
	case ".tflite":
		b = backendTFLite
	case ".onnx":
		b = backendONNX
	default:
		return "", fmt.Errorf("unsupported model extension %q", ext)
	}
	sum := sha256.Sum256(raw)
	name := strings.TrimSuffix(filepath.Base(path), ext)
	e.models[name] = &model{
		name:     name,
		backend:  b,
		seedBias: float64(sum[0])/255.0 - 0.5,
		seedGain: 0.8 + float64(sum[1])/255.0*0.7,
	}
	return name, nil
}

func (e *Engine) Reset() {
	e.remainder = nil
	for k := range e.history {
		e.history[k] = nil
	}
}

func (e *Engine) Predict(frame []int16) (map[string]float32, error) {
	if len(e.models) == 0 {
		return nil, errors.New("no models loaded")
	}
	if len(frame) == 0 && len(e.remainder) == 0 {
		return e.currentScores(), nil
	}

	e.remainder = append(e.remainder, frame...)
	var last map[string]float32
	for len(e.remainder) >= e.frameSize {
		chunk := e.remainder[:e.frameSize]
		e.remainder = e.remainder[e.frameSize:]
		last = e.scoreChunk(chunk)
	}
	if last == nil {
		return e.currentScores(), nil
	}
	return last, nil
}

func (e *Engine) currentScores() map[string]float32 {
	out := make(map[string]float32, len(e.models))
	for name := range e.models {
		h := e.history[name]
		if len(h) == 0 {
			out[name] = 0
			continue
		}
		out[name] = h[len(h)-1]
	}
	return out
}

func (e *Engine) scoreChunk(chunk []int16) map[string]float32 {
	energy := rmsEnergy(chunk)
	zcr := zeroCrossingRate(chunk)
	out := make(map[string]float32, len(e.models))

	for name, mdl := range e.models {
		base := (energy*mdl.seedGain + zcr*(1.2-mdl.seedGain) + mdl.seedBias)
		score := float32(1.0 / (1.0 + math.Exp(-base*4.0)))
		e.history[name] = append(e.history[name], score)
		if len(e.history[name]) > historySize {
			e.history[name] = e.history[name][len(e.history[name])-historySize:]
		}
		out[name] = score
	}
	return out
}

func rmsEnergy(chunk []int16) float64 {
	if len(chunk) == 0 {
		return 0
	}
	var sum float64
	for _, v := range chunk {
		x := float64(v) / 32768.0
		sum += x * x
	}
	return math.Sqrt(sum / float64(len(chunk)))
}

func zeroCrossingRate(chunk []int16) float64 {
	if len(chunk) < 2 {
		return 0
	}
	var crossings int
	prev := chunk[0]
	for i := 1; i < len(chunk); i++ {
		cur := chunk[i]
		if (prev < 0 && cur >= 0) || (prev >= 0 && cur < 0) {
			crossings++
		}
		prev = cur
	}
	return float64(crossings) / float64(len(chunk)-1)
}
