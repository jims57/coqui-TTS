import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import javax.sound.sampled.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.List;

public class Main {
    // 使用原始 URL 但尝试不同的方法
    private static final String SERVER_URL = "ws://47.116.221.45:9002/aqs-tts";
    private static final String API_KEY = "sk-5z6y7x8w9v0u1t2s3r4q5p6o7n8m9l0k1j2i3h4g";
    private static final String OUTPUT_DIR = "audio_output";
    
    // 使用 CountDownLatch 确保我们等待所有数据块
    private static final CountDownLatch completionLatch = new CountDownLatch(1);
    
    public static void main(String[] args) {
        try {
            // 如果输出目录不存在，则创建它
            Path outputPath = Paths.get(OUTPUT_DIR);
            if (!Files.exists(outputPath)) {
                Files.createDirectory(outputPath);
            }
            
            // 定义请求参数
            // String text = "Hello, this is a test of the Coqui TTS API. The quick brown fox jumps over the lazy dog.";
            String text = "昨天我在书店发现了一本很有趣的小说，立刻就买下来了。";
            String language = "zh";
            String audioFormat = "wav";
            boolean saveAudioFile = true;
            int speakerId = 1;
            double speed = 1.0;
            
            // 创建 JSON 请求
            String request = String.format(
                "{\"text\":\"%s\",\"language\":\"%s\",\"audioFormat\":\"%s\",\"saveAudioFile\":%b,\"speakerId\":%d,\"speed\":%f}",
                escapeJson(text), language, audioFormat, saveAudioFile, speakerId, speed
            );
            
            System.out.println("Request: " + request);
            
            // 创建自定义流式音频处理程序以捕获所有块
            StreamingAudioHandler audioHandler = new StreamingAudioHandler();
            
            // 设置具有增强配置的 WebSocket 客户端，以实现可靠的流传输
            WebSocket webSocket = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(60))
                .build()
                .newWebSocketBuilder()
                .header("X-API-Key", API_KEY)
                .buildAsync(URI.create(SERVER_URL + "?api_key=" + API_KEY), audioHandler)
                .join();
            
            // 发送请求生成音频
            System.out.println("Sending request to server...");
            webSocket.sendText(request, true);
            
            // 等待接收所有音频数据
            System.out.println("Waiting for audio streaming to complete...");
            boolean completed = completionLatch.await(2, TimeUnit.MINUTES);
            
            if (!completed) {
                System.out.println("WARNING: Timed out waiting for audio streaming to complete");
            } else {
                System.out.println("Audio streaming completed successfully");
            }
            
            // 处理并播放完整音频
            audioHandler.processCompleteAudio();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 辅助方法，用于转义 JSON 字符串中的特殊字符
    private static String escapeJson(String input) {
        if (input == null) return "";
        
        StringBuilder escaped = new StringBuilder();
        for (char c : input.toCharArray()) {
            switch (c) {
                case '\"': escaped.append("\\\""); break;
                case '\\': escaped.append("\\\\"); break;
                case '/': escaped.append("\\/"); break;
                case '\b': escaped.append("\\b"); break;
                case '\f': escaped.append("\\f"); break;
                case '\n': escaped.append("\\n"); break;
                case '\r': escaped.append("\\r"); break;
                case '\t': escaped.append("\\t"); break;
                default: escaped.append(c);
            }
        }
        return escaped.toString();
    }

    static class StreamingAudioHandler implements WebSocket.Listener {
        private final List<byte[]> audioChunks = new ArrayList<>();
        private final String timestamp = String.valueOf(System.currentTimeMillis());
        private boolean isFirstChunk = true;
        private boolean receivedEmptyChunk = false;
        private boolean headerProcessed = false;
        private byte[] headerData = null;
        private int chunkCount = 0;
        private long startTime = System.currentTimeMillis();
        private long totalBytesReceived = 0;
        private AudioFormat audioFormat = null;
        private int audioDataStart = 0;
        
        @Override
        public void onOpen(WebSocket webSocket) {
            System.out.println("WebSocket connection established");
            // 请求大量消息以确保我们收到所有内容
            webSocket.request(1000);
        }
        
        @Override
        public CompletionStage<?> onBinary(WebSocket webSocket, ByteBuffer data, boolean last) {
            byte[] bytes = new byte[data.remaining()];
            data.get(bytes);
            
            chunkCount++;
            
            if (bytes.length > 0) {
                if (isFirstChunk) {
                    isFirstChunk = false;
                    long firstChunkTime = System.currentTimeMillis() - startTime;
                    System.out.println("Received first audio chunk after " + firstChunkTime + " ms");
                    
                    // 检查这是否是 WAV 头
                    if (bytes.length > 12 && 
                        bytes[0] == 'R' && bytes[1] == 'I' && bytes[2] == 'F' && bytes[3] == 'F' &&
                        bytes[8] == 'W' && bytes[9] == 'A' && bytes[10] == 'V' && bytes[11] == 'E') {
                        
                        System.out.println("Detected WAV header in first chunk");
                        
                        // 存储头数据
                        headerData = bytes.clone();
                        headerProcessed = true;
                        
                        // 从头部提取格式信息
                        extractWavFormat(bytes);
                    }
                }
                
                // 存储此块
                audioChunks.add(bytes);
                totalBytesReceived += bytes.length;
                
                System.out.println("Received chunk #" + chunkCount + ": " + bytes.length + 
                                  " bytes (Total: " + totalBytesReceived + " bytes)");
                
                // 每 3 个块保存一次中间文件以进行监控
                if (chunkCount % 3 == 0) {
                    saveIntermediateFile();
                }
            } else {
                System.out.println("Received empty chunk - may indicate end of stream");
                receivedEmptyChunk = true;
                
                // 当我们获得空块时发出完成信号
                completionLatch.countDown();
            }
            
            if (last) {
                System.out.println("Server indicated this is the LAST chunk - stream complete");
                // 首先等待空块或关闭事件
            }
            
            // 继续请求数据，直到我们得到一个空块
            if (!receivedEmptyChunk) {
                webSocket.request(10);
            } else {
                // 仅在收到空块时关闭
                webSocket.sendClose(WebSocket.NORMAL_CLOSURE, "Audio stream complete");
            }
            
            return CompletableFuture.completedFuture(null);
        }
        
        private void extractWavFormat(byte[] wavHeader) {
            try {
                // Print the raw byte values for channels (bytes 22-23)
                System.out.println("Raw channel bytes: wavHeader[22]=" + (wavHeader[22] & 0xFF) + 
                                  ", wavHeader[23]=" + (wavHeader[23] & 0xFF));

                // 从 WAV 头中提取基本格式信息
                // Extract number of audio channels from bytes 22-23
                // WAV format stores this as a 16-bit value in little-endian order
                // We mask each byte with 0xFF to ensure we only get the unsigned byte value
                // Then combine them with bitwise OR after shifting the second byte
                int channels = (wavHeader[22] & 0xFF) | ((wavHeader[23] & 0xFF) << 8);
                
                // Extract sample rate from bytes 24-27
                // WAV format stores this as a 32-bit value in little-endian order
                // We read 4 bytes, masking each with 0xFF, then shift and combine with bitwise OR
                // Byte 24 is least significant, byte 27 is most significant
                int sampleRate = (wavHeader[24] & 0xFF) | ((wavHeader[25] & 0xFF) << 8) | 
                               ((wavHeader[26] & 0xFF) << 16) | ((wavHeader[27] & 0xFF) << 24);
                
                // Extract bits per sample from bytes 34-35
                // WAV format stores this as a 16-bit value in little-endian order
                // Common values are 8, 16, 24, or 32 bits per sample
                int bitsPerSample = (wavHeader[34] & 0xFF) | ((wavHeader[35] & 0xFF) << 8);
                
                System.out.println("WAV format: " + sampleRate + " Hz, " + bitsPerSample + 
                                  " bits, " + channels + " channels");
                
                // 创建 AudioFormat 对象
                boolean signed = true;
                boolean bigEndian = false;  // WAV 通常是小端序
                audioFormat = new AudioFormat(sampleRate, bitsPerSample, channels, 
                                           signed, bigEndian);
                
                // 查找音频数据的起始位置（在"data"块之后）
                for (int i = 12; i < wavHeader.length - 8; i++) {
                    if (wavHeader[i] == 'd' && wavHeader[i+1] == 'a' && 
                        wavHeader[i+2] == 't' && wavHeader[i+3] == 'a') {
                        audioDataStart = i + 8;  // 跳过 "data" + 4 字节大小
                        System.out.println("Audio data starts at offset: " + audioDataStart);
                        break;
                    }
                }
            } catch (Exception e) {
                System.err.println("Error extracting WAV format: " + e.getMessage());
            }
        }
        
        private void saveIntermediateFile() {
            try {
                ByteArrayOutputStream combined = new ByteArrayOutputStream();
                
                // 合并到目前为止收到的所有块
                for (byte[] chunk : audioChunks) {
                    combined.write(chunk);
                }
                
                // 保存中间 WAV 文件
                String intermediateFile = OUTPUT_DIR + "/streaming_" + timestamp + "_progress.wav";
                Files.write(Paths.get(intermediateFile), combined.toByteArray());
                
                System.out.println("Saved intermediate file with " + totalBytesReceived + " bytes");
            } catch (IOException e) {
                System.err.println("Error saving intermediate file: " + e.getMessage());
            }
        }
        
        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            System.out.println("Received text message: " + data);
            webSocket.request(1);
            return CompletableFuture.completedFuture(null);
        }
        
        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            System.out.println("WebSocket closed with status " + statusCode + ": " + reason);
            
            // 在 WebSocket 关闭时始终发出完成信号
            // 这确保我们不会挂起等待可能永远不会到来的空块
            completionLatch.countDown();
            
            return CompletableFuture.completedFuture(null);
        }
        
        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            System.err.println("WebSocket error: " + error.getMessage());
            
            // 即使出错也发出完成信号
            completionLatch.countDown();
        }
        
        public void processCompleteAudio() {
            try {
                if (audioChunks.isEmpty()) {
                    System.err.println("No audio chunks received");
                    return;
                }
                
                System.out.println("Processing " + audioChunks.size() + " audio chunks");
                
                // 更改 1：直到收到空块或 WebSocket 关闭后再处理
                // 这确保我们在处理前等待所有块
                if (!receivedEmptyChunk) {
                    System.out.println("Warning: Processing audio before receiving empty chunk marker");
                }
                
                // 合并所有音频块
                ByteArrayOutputStream completeAudio = new ByteArrayOutputStream();
                
                // 更改 2：正确处理音频块以消除"dada"声音
                // 类似于 iOS 处理方式
                if (headerProcessed && audioChunks.size() > 0) {
                    // 第一个块（包含 WAV 头）- 完全写入
                    completeAudio.write(audioChunks.get(0));
                    
                    // 在第一个块中查找音频数据起始位置（WAV 头之后）
                    int headerSize = findDataChunkStart(audioChunks.get(0));
                    
                    // 对于后续块，如果存在，则跳过它们的 WAV 头
                    for (int i = 1; i < audioChunks.size(); i++) {
                        byte[] chunk = audioChunks.get(i);
                        
                        // 如果块以"RIFF"开头 - 这是一个新的 WAV 文件，跳过它的头
                        if (chunk.length > 12 && 
                            chunk[0] == 'R' && chunk[1] == 'I' && chunk[2] == 'F' && chunk[3] == 'F' &&
                            chunk[8] == 'W' && chunk[9] == 'A' && chunk[10] == 'V' && chunk[11] == 'E') {
                            
                            // 在此块中查找数据块起始位置
                            int chunkDataStart = findDataChunkStart(chunk);
                            
                            // 仅写入音频数据部分（跳过头）
                            if (chunkDataStart > 0 && chunkDataStart < chunk.length) {
                                completeAudio.write(chunk, chunkDataStart, chunk.length - chunkDataStart);
                                System.out.println("Chunk #" + (i+1) + ": Skipped " + chunkDataStart + 
                                                 " bytes of WAV header to avoid artifacts");
                            } else {
                                // 如果我们找不到数据块，就按原样追加块
                                completeAudio.write(chunk);
                            }
                        } else {
                            // 常规块 - 按原样写入
                            completeAudio.write(chunk);
                        }
                    }
                } else {
                    // 如果头处理失败 - 只需连接所有块
                    for (byte[] chunk : audioChunks) {
                        completeAudio.write(chunk);
                    }
                }
                
                byte[] fullAudioData = completeAudio.toByteArray();
                
                // 如果我们需要修复 WAV 头
                if (headerProcessed && headerData != null) {
                    fullAudioData = fixWavHeader(fullAudioData);
                }
                
                // 保存完整的音频文件
                String outputFile = OUTPUT_DIR + "/complete_" + timestamp + ".wav";
                Files.write(Paths.get(outputFile), fullAudioData);
                System.out.println("Saved complete audio file: " + outputFile);
                
                // 同时保存为"latest.wav"以便更容易访问
                Files.write(Paths.get(OUTPUT_DIR + "/latest.wav"), fullAudioData);
                
                // 播放音频
                playCompleteAudio(outputFile);
                
            } catch (Exception e) {
                System.err.println("Error processing complete audio: " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        private byte[] fixWavHeader(byte[] audioData) {
            try {
                // 仅在这看起来像 WAV 文件时修复
                if (audioData.length < 44 || audioData[0] != 'R' || audioData[1] != 'I' || 
                    audioData[2] != 'F' || audioData[3] != 'F') {
                    System.err.println("Not a valid WAV file, can't fix header");
                    return audioData;
                }
                
                System.out.println("Fixing WAV header for " + audioData.length + " bytes of audio data");
                
                // 更新头中的文件大小（总大小 - 8 字节）
                int fileSize = audioData.length - 8;
                audioData[4] = (byte)(fileSize & 0xFF);
                audioData[5] = (byte)((fileSize >> 8) & 0xFF);
                audioData[6] = (byte)((fileSize >> 16) & 0xFF);
                audioData[7] = (byte)((fileSize >> 24) & 0xFF);
                
                // 查找数据块并更新其大小
                for (int i = 36; i < Math.min(audioData.length - 4, 100); i++) {
                    if (audioData[i] == 'd' && audioData[i+1] == 'a' && 
                        audioData[i+2] == 't' && audioData[i+3] == 'a') {
                        
                        // 在位置 i 找到数据块，在 i+4 更新大小
                        int dataSize = audioData.length - (i + 8);
                        audioData[i+4] = (byte)(dataSize & 0xFF);
                        audioData[i+5] = (byte)((dataSize >> 8) & 0xFF);
                        audioData[i+6] = (byte)((dataSize >> 16) & 0xFF);
                        audioData[i+7] = (byte)((dataSize >> 24) & 0xFF);
                        
                        System.out.println("Updated data chunk size to " + dataSize + " bytes");
                        break;
                    }
                }
                
                return audioData;
            } catch (Exception e) {
                System.err.println("Error fixing WAV header: " + e.getMessage());
                return audioData; // 出错时返回原始数据
            }
        }
        
        private void playCompleteAudio(String filePath) {
            try {
                File audioFile = new File(filePath);
                AudioInputStream audioStream = AudioSystem.getAudioInputStream(audioFile);
                
                AudioFormat format = audioStream.getFormat();
                System.out.println("Audio format: " + format);
                
                // 计算持续时间
                long frames = audioStream.getFrameLength();
                double durationInSeconds = (double)frames / format.getFrameRate();
                System.out.println("Audio duration: " + durationInSeconds + " seconds");
                System.out.println("Audio frames: " + frames);
                
                // 播放音频
                DataLine.Info info = new DataLine.Info(SourceDataLine.class, format);
                
                if (!AudioSystem.isLineSupported(info)) {
                    System.err.println("Line not supported for format: " + format);
                    return;
                }
                
                SourceDataLine line = (SourceDataLine) AudioSystem.getLine(info);
                line.open(format);
                line.start();
                
                System.out.println("Playing complete audio...");
                
                // 使用大缓冲区以获得更好的性能
                byte[] buffer = new byte[8192];
                int bytesRead;
                
                while ((bytesRead = audioStream.read(buffer, 0, buffer.length)) != -1) {
                    line.write(buffer, 0, bytesRead);
                }
                
                line.drain();
                line.close();
                audioStream.close();
                
                System.out.println("Audio playback complete");
                
            } catch (Exception e) {
                System.err.println("Error playing audio: " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        private int findDataChunkStart(byte[] wavData) {
            // 查找数据块的起始位置
            for (int i = 12; i < wavData.length - 8; i++) {
                if (wavData[i] == 'd' && wavData[i+1] == 'a' && 
                    wavData[i+2] == 't' && wavData[i+3] == 'a') {
                    return i + 8;  // 跳过 "data" + 4 字节大小
                }
            }
            // 如果找不到数据块，则默认为标准 WAV 头大小
            return 44;
        }
    }
}

