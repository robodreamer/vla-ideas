# OFT-inspired systems toy report

Seed: `17`. Trials per method: `96`.

This generated result is a deterministic delayed-observation 2D systems toy, not an OpenVLA-OFT reproduction.

| Method | latency (ms, simulated) | action-output throughput (Hz) | mean tracking error | success | jerk proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| autoregressive_serial | 55 | 12.5 | 0.334 | 29.2% | 32.9 |
| parallel_open_loop | 60 | 25.6 | 0.332 | 28.1% | 15.2 |
| parallel_refresh_4 | 60 | 50.0 | 0.316 | 33.3% | 19.0 |
| parallel_refresh_2 | 60 | 100.0 | 0.311 | 28.1% | 20.0 |

Interpretation: serial decoding has an action-rate penalty; full open-loop chunks preserve rate but amplify delayed-state error. More frequent closed-loop refresh trades extra parallel decisions for better tracking, while action discontinuities can raise the jerk proxy.

Only the systems hypotheses are tested: temporal output interface, chunk horizon, refresh cadence, and delayed feedback. No VLM, robot, OpenVLA weights, benchmark, training data, or paper numerical claim is represented.
