#pragma once

#include <memory>

// Simple tonal synth — 4 sine oscillators with exponential decay envelopes.
// Plays a major chord spanning one octave (C4, E4, G4, C5).
// Used for audible feedback when sequencer steps fire.

class Synth {
public:
  Synth();
  ~Synth();

  Synth(const Synth&) = delete;
  Synth& operator=(const Synth&) = delete;

  void init();
  void deinit();

  // Trigger a pluck on the given channel (0–3). Thread-safe.
  void trigger(int channel);

  // Enable/disable audio output.
  void set_enabled(bool enabled);
  bool is_enabled() const;

  // Gain: 0.0 = silent, 1.0 = full volume
  void set_gain(float gain);
  float gain() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
