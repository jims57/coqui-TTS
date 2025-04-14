Opus:  Raw Audio → Encode → Network Transfer (faster) → Decode → Playback

encoder = Encoder(SAMPLE_RATE, CHANNELS, 'audio')
# Optimize for voice
encoder.set_bitrate(32000)  # 32kbps is often enough for voice
encoder.set_vbr(True)
encoder.set_packet_loss_perc(5)  # More resilient to packet loss


Recommendation: Use Opus
The encoding/decoding overhead (~100ms) is negligible compared to the network transfer time savings
Smaller packets have better chances of getting through network congestion
Less data transfer means fewer potential packet losses and retransmissions
More resilient to unstable connections
For AWS → China connections, network transfer is definitely your bottleneck, not processing time. The 6x smaller size of Opus will give you much better real-world performance despite the small encoding/decoding overhead.


Recommendation:
Stick with Opus because:
It offers the best balance of size and quality
Modern and well-supported
Low latency
While AMR-WB might be smaller, the quality trade-off is not worth it
Opus already uses state-of-the-art compression techniques
If you really need to optimize Opus size further, you can tune its parameters:


WAV (uncompressed): 390KB
Opus: 66KB
AAC-HE: ~70-80KB
AMR-WB: ~45-55KB
Speex: ~60-70KB


If you really need to optimize Opus size further, you can tune its parameters:
encoder = Encoder(SAMPLE_RATE, CHANNELS, 'audio')
# Optimize for minimum size while maintaining acceptable quality
encoder.set_bitrate(24000)  # Lower bitrate (24kbps)
encoder.set_vbr(True)
encoder.set_complexity(5)  # Lower complexity (range 0-10)
encoder.set_signal_type(OpusSignal.OPUS_SIGNAL_VOICE)  # Optimize for voice

However, going below Opus's standard settings might start affecting quality noticeably. Opus is already highly optimized for voice content, and it's unlikely you'll find a significantly smaller format that maintains acceptable quality.