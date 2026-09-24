// motion.peak_decay — helpers shared by meter.hlsl / apply.hlsl. The decay
// gain must be computed identically in both passes (the meter's Rise Only
// takeover compares against the same decaying level the apply pass shows).

float pd_lum(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// 1 through `hold`, then a smoothstep sigmoid down to 1-amount over `fall`
// (fall = 0 is a hard cliff; the epsilon keeps the edges ordered).
float pd_gain(float age, float amount, float hold, float fall) {
  return 1.0 - amount * smoothstep(hold, hold + max(fall, 0.001), age);
}
