TCP Handshake Explained
The TCP handshake (also called 3-way handshake) is a process that happens before any HTTP data can be sent. It establishes a reliable connection between the client and server. In your image, it took 318ms.
The process involves 3 steps:
SYN: Client sends a SYN (synchronize) packet to the server
SYN-ACK: Server responds with SYN-ACK packet
ACK: Client sends an ACK (acknowledge) packet back


        Client          Server
        |     SYN      |
        |------------->|
        |   SYN-ACK   |
        |<------------|
        |     ACK     |
        |------------->|

# Current solution(http):
1. TCP Handshake: 1 RTT (318ms in your case)
2. TLS Handshake: 1 RTT (typically ~100-200ms)
-------------------------------------------

# QUICK(Http3) solution:
1. Combined crypto + transport handshake: 1 RTT
2. With 0-RTT resumption: 0 RTT for repeat connections
-------------------------------------------
Estimated reduction:
- First connection: ~50% reduction (200-250ms instead of 400-500ms)
- Repeat connections: ~90% reduction (near 0ms with 0-RTT resumption)


# Estimate improvement
First Connection:
- Current: ~400-500ms
- With QUIC: ~200-250ms
- Reduction: 200-250ms (50%)

Repeat Connections:
- Current: ~400-500ms
- With QUIC + 0-RTT: ~0-50ms
- Reduction: 350-450ms (90%)
