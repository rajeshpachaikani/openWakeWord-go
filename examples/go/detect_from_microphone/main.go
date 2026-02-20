package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gen2brain/malgo"
	openwakeword "github.com/rajeshpachaikani/openWakeWord-go"
)

const (
	sampleRate = 16000
	frameSize  = 1280
	audioQueue = 32
)

func main() {
	var (
		modelPathsCSV = flag.String("models", "", "Comma-separated model file paths (.onnx or .tflite)")
		threshold     = flag.Float64("threshold", 0.55, "Detection threshold [0.0-1.0]")
		cooldown      = flag.Duration("cooldown", 1500*time.Millisecond, "Minimum time between detections per model")
	)
	flag.Parse()

	engine := openwakeword.NewEngine(openwakeword.Options{
		SampleRate: sampleRate,
		FrameSize:  frameSize,
	})

	modelPaths := splitAndTrim(*modelPathsCSV)
	if len(modelPaths) == 0 {
		log.Fatal("no models provided; pass at least one path via -models")
	}

	for _, modelPath := range modelPaths {
		name, err := engine.AddModel(modelPath)
		if err != nil {
			log.Fatalf("failed to load model %q: %v", modelPath, err)
		}
		log.Printf("loaded model %q from %q", name, modelPath)
	}

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, func(message string) {
		if strings.TrimSpace(message) != "" {
			log.Printf("audio backend: %s", strings.TrimSpace(message))
		}
	})
	if err != nil {
		log.Fatalf("failed to initialize audio context: %v", err)
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.SampleRate = sampleRate
	deviceConfig.Capture.Format = malgo.FormatS16
	deviceConfig.Capture.Channels = 1
	deviceConfig.Alsa.NoMMap = 1

	audioFrames := make(chan []int16, audioQueue)

	callbacks := malgo.DeviceCallbacks{
		Data: func(_ []byte, inputSamples []byte, _ uint32) {
			if len(inputSamples) == 0 {
				return
			}
			sampleCount := len(inputSamples) / 2
			pcm := make([]int16, sampleCount)
			for i := 0; i < sampleCount; i++ {
				pcm[i] = int16(binary.LittleEndian.Uint16(inputSamples[i*2 : i*2+2]))
			}
			select {
			case audioFrames <- pcm:
			default:
				// Drop when full to avoid blocking the realtime audio callback.
			}
		},
	}

	device, err := malgo.InitDevice(ctx.Context, deviceConfig, callbacks)
	if err != nil {
		log.Fatalf("failed to initialize capture device: %v", err)
	}
	defer device.Uninit()

	if err := device.Start(); err != nil {
		log.Fatalf("failed to start capture device: %v", err)
	}

	log.Printf("listening on microphone @ %d Hz (mono). threshold=%.3f", sampleRate, *threshold)
	log.Printf("say your wakeword now; press Ctrl+C to stop")

	lastDetectionAt := map[string]time.Time{}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	for {
		select {
		case <-stop:
			fmt.Println()
			log.Println("stopping detector")
			return
		case frame := <-audioFrames:
			scores, err := engine.Predict(frame)
			if err != nil {
				log.Printf("predict error: %v", err)
				continue
			}
			now := time.Now()
			for modelName, score := range scores {
				if float64(score) < *threshold {
					continue
				}
				if now.Sub(lastDetectionAt[modelName]) < *cooldown {
					continue
				}
				lastDetectionAt[modelName] = now
				log.Printf("WAKEWORD DETECTED model=%s score=%.4f", modelName, score)
			}
		}
	}
}

func splitAndTrim(csv string) []string {
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		v := strings.TrimSpace(part)
		if v != "" {
			out = append(out, v)
		}
	}
	return out
}
