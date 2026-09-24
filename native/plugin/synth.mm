#include "synth.h"

#import <AVFoundation/AVFoundation.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

// Major chord spanning one octave: C5, E5, G5, C6
static const float kFrequencies[4] = {
  523.25f, // C5
  659.26f, // E5
  783.99f, // G5
  1046.50f, // C6
};

// Envelope: exponential decay from 1.0 to ~0.001 in ~200ms at 44100Hz
// decay = pow(0.001, 1 / (0.2 * 44100)) ≈ 0.99922
static const float kDecayPerSample = 0.99922f;
static const float kEnvelopeThreshold = 0.001f;
static const float kGain = 1.0f; // master volume

struct Voice {
  float phase = 0;
  float envelope = 0;
  float frequency = 0;
};

struct Synth::Impl {
  AVAudioEngine* engine = nil;
  AVAudioSourceNode* srcNode = nil;
  std::atomic<bool> enabled{false};
  std::atomic<float> gain{0.5f};

  Voice voices[4];
  std::atomic<bool> pending[4] = {};

  void renderToBufferList(AudioBufferList* abl, int frameCount, float sampleRate) {
    // Check for pending triggers
    for (int ch = 0; ch < 4; ++ch) {
      if (pending[ch].exchange(false)) {
        voices[ch].envelope = 1.0f;
        voices[ch].phase = 0;
        voices[ch].frequency = kFrequencies[ch];
      }
    }

    // Clear all output buffers
    for (UInt32 buf = 0; buf < abl->mNumberBuffers; ++buf) {
      memset(abl->mBuffers[buf].mData, 0, abl->mBuffers[buf].mDataByteSize);
    }

    float g = gain.load();

    // Render voices into a temp mono buffer, then copy to all output buffers
    std::vector<float> mono(frameCount, 0);
    for (int ch = 0; ch < 4; ++ch) {
      auto& v = voices[ch];
      if (v.envelope < kEnvelopeThreshold) continue;

      float phaseInc = v.frequency * 2.0f * M_PI / sampleRate;
      for (int i = 0; i < frameCount; ++i) {
        mono[i] += sinf(v.phase) * v.envelope * g * kGain;
        v.phase += phaseInc;
        v.envelope *= kDecayPerSample;
      }
      if (v.phase > 2.0f * M_PI * 1000)
        v.phase = fmodf(v.phase, 2.0f * M_PI);
      if (v.envelope < kEnvelopeThreshold)
        v.envelope = 0;
    }

    // Copy mono mix to all output buffers (handles both interleaved and non-interleaved)
    for (UInt32 buf = 0; buf < abl->mNumberBuffers; ++buf) {
      float* out = (float*)abl->mBuffers[buf].mData;
      int channels = abl->mBuffers[buf].mNumberChannels;
      for (int i = 0; i < frameCount; ++i) {
        for (int c = 0; c < channels; ++c) {
          out[i * channels + c] = mono[i];
        }
      }
    }
  }
};

Synth::Synth() : impl_(std::make_unique<Impl>()) {}
Synth::~Synth() { deinit(); }

void Synth::init() {
  @autoreleasepool {
    impl_->engine = [[AVAudioEngine alloc] init];

    auto* rawImpl = impl_.get();

    AVAudioFormat* format = [[AVAudioFormat alloc]
        initStandardFormatWithSampleRate:44100 channels:2];

    impl_->srcNode = [[AVAudioSourceNode alloc]
        initWithFormat:format
        renderBlock:^OSStatus(BOOL* _Nonnull isSilence,
                              const AudioTimeStamp* _Nonnull timestamp,
                              AVAudioFrameCount frameCount,
                              AudioBufferList* _Nonnull outputData) {
      if (!rawImpl->enabled.load()) {
        *isSilence = YES;
        return noErr;
      }
      rawImpl->renderToBufferList(outputData, (int)frameCount, 44100.0f);
      return noErr;
    }];

    [impl_->engine attachNode:impl_->srcNode];
    [impl_->engine connect:impl_->srcNode
                        to:impl_->engine.mainMixerNode
                    format:format];

    NSError* error = nil;
    [impl_->engine startAndReturnError:&error];
  }
}

void Synth::deinit() {
  @autoreleasepool {
    if (impl_->engine) {
      [impl_->engine stop];
      impl_->engine = nil;
      impl_->srcNode = nil;
    }
  }
}

void Synth::trigger(int channel) {
  if (channel >= 0 && channel < 4)
    impl_->pending[channel].store(true);
}

void Synth::set_enabled(bool enabled) {
  impl_->enabled.store(enabled);
}

bool Synth::is_enabled() const {
  return impl_->enabled.load();
}

void Synth::set_gain(float gain) {
  impl_->gain.store(gain);
}

float Synth::gain() const {
  return impl_->gain.load();
}
