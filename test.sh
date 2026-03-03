./build/bin/whisper-cli -m models/ggml-large-v3-turbo-q4_0.bin -f test.wav

# CPU+GPU+ANE Power: 3.27W (avg: 18.06W peak: 26.71W) throttle: no

# whisper_print_timings:     load time =  1172.48 ms
# whisper_print_timings:     fallbacks =  20 p / 122 h
# whisper_print_timings:      mel time =   741.96 ms
# whisper_print_timings:   sample time = 14774.76 ms / 60071 runs (     0.25 ms per run)
# whisper_print_timings:   encode time = 53447.62 ms /    93 runs (   574.71 ms per run)
# whisper_print_timings:   decode time =  4041.63 ms /  2148 runs (     1.88 ms per run)
# whisper_print_timings:   batchd time = 36094.60 ms / 57394 runs (     0.63 ms per run)
# whisper_print_timings:   prompt time =  2267.35 ms / 22938 runs (     0.10 ms per run)
# whisper_print_timings:    total time =