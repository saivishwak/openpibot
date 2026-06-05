package com.xlerobot.questteleop;

import android.graphics.Rect;
import android.graphics.ImageFormat;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;

import java.io.ByteArrayOutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.nio.ByteBuffer;
import java.util.Arrays;

public final class RtpH264Receiver {
    private static final byte[] START_CODE = new byte[] {0, 0, 0, 1};

    private final Object frameLock = new Object();
    private volatile boolean running;
    private volatile String lastError = "";
    private volatile int frameWidth;
    private volatile int frameHeight;
    private volatile int frameCount;

    private DatagramSocket socket;
    private MediaCodec decoder;
    private Thread receiveThread;
    private Thread drainThread;
    private byte[] latestRgba;
    private byte[] fuBuffer;
    private int fuLength;

    public void start(int port, int width, int height) {
        stop();
        frameWidth = Math.max(16, width);
        frameHeight = Math.max(16, height);
        running = true;
        lastError = "";
        frameCount = 0;
        receiveThread = new Thread(() -> runReceiver(port), "xlerobot-rtp-h264-rx-" + port);
        receiveThread.start();
    }

    public void stop() {
        running = false;
        try {
            if (socket != null) {
                socket.close();
            }
        } catch (Exception ignored) {
        }
        socket = null;
        try {
            if (decoder != null) {
                decoder.stop();
                decoder.release();
            }
        } catch (Exception ignored) {
        }
        decoder = null;
    }

    public boolean isRunning() {
        return running;
    }

    public int getFrameCount() {
        return frameCount;
    }

    public String getLastError() {
        return lastError;
    }

    public int getFrameWidth() {
        return frameWidth;
    }

    public int getFrameHeight() {
        return frameHeight;
    }

    public byte[] getLatestRgba() {
        synchronized (frameLock) {
            return latestRgba == null ? null : latestRgba.clone();
        }
    }

    private void runReceiver(int port) {
        try {
            decoder = MediaCodec.createDecoderByType("video/avc");
            MediaFormat format = MediaFormat.createVideoFormat("video/avc", frameWidth, frameHeight);
            format.setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            decoder.configure(format, null, null, 0);
            decoder.start();
            drainThread = new Thread(this::drainDecoder, "xlerobot-h264-drain");
            drainThread.start();

            socket = new DatagramSocket(port);
            socket.setReceiveBufferSize(2 * 1024 * 1024);
            byte[] buffer = new byte[65535];
            DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
            while (running) {
                socket.receive(packet);
                processRtpPacket(packet.getData(), packet.getLength());
            }
        } catch (Exception e) {
            if (running) {
                lastError = e.getClass().getSimpleName() + ": " + e.getMessage();
            }
        } finally {
            running = false;
        }
    }

    private void drainDecoder() {
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        while (running && decoder != null) {
            try {
                int index = decoder.dequeueOutputBuffer(info, 10000);
                if (index >= 0) {
                    try {
                        if ((info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && info.size > 0) {
                            drainOutputBuffer(index);
                        }
                    } finally {
                        decoder.releaseOutputBuffer(index, false);
                    }
                } else if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat format = decoder.getOutputFormat();
                    updateFrameSize(format);
                }
            } catch (Exception e) {
                if (running) {
                    lastError = "decoder drain: " + e.getMessage();
                }
                return;
            }
        }
    }

    private void drainOutputBuffer(int index) {
        Image image = null;
        try {
            image = decoder.getOutputImage(index);
            if (image != null) {
                byte[] rgba = yuv420ToRgba(image);
                if (rgba != null && rgba.length > 0) {
                    synchronized (frameLock) {
                        latestRgba = rgba;
                    }
                    frameCount += 1;
                }
            } else {
                lastError = "decoder output image unavailable";
            }
        } catch (Exception e) {
            lastError = "image conversion: " + e.getMessage();
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    private void updateFrameSize(MediaFormat format) {
        if (format == null) {
            return;
        }
        int width = format.containsKey(MediaFormat.KEY_WIDTH)
                ? format.getInteger(MediaFormat.KEY_WIDTH)
                : frameWidth;
        int height = format.containsKey(MediaFormat.KEY_HEIGHT)
                ? format.getInteger(MediaFormat.KEY_HEIGHT)
                : frameHeight;
        int cropLeft = format.containsKey("crop-left") ? format.getInteger("crop-left") : 0;
        int cropRight = format.containsKey("crop-right") ? format.getInteger("crop-right") : width - 1;
        int cropTop = format.containsKey("crop-top") ? format.getInteger("crop-top") : 0;
        int cropBottom = format.containsKey("crop-bottom") ? format.getInteger("crop-bottom") : height - 1;
        frameWidth = Math.max(1, cropRight - cropLeft + 1);
        frameHeight = Math.max(1, cropBottom - cropTop + 1);
    }

    private void processRtpPacket(byte[] packet, int length) throws Exception {
        if (length < 13 || decoder == null) {
            return;
        }
        int cc = packet[0] & 0x0F;
        boolean extension = (packet[0] & 0x10) != 0;
        int offset = 12 + cc * 4;
        if (offset >= length) {
            return;
        }
        if (extension) {
            if (offset + 4 > length) {
                return;
            }
            int extWords = ((packet[offset + 2] & 0xFF) << 8) | (packet[offset + 3] & 0xFF);
            offset += 4 + extWords * 4;
            if (offset >= length) {
                return;
            }
        }

        int nalType = packet[offset] & 0x1F;
        if (nalType >= 1 && nalType <= 23) {
            queueAccessUnit(packet, offset, length - offset);
        } else if (nalType == 24) {
            processStapA(packet, offset + 1, length);
        } else if (nalType == 28) {
            processFuA(packet, offset, length);
        }
    }

    private void processStapA(byte[] packet, int offset, int length) throws Exception {
        while (offset + 2 <= length) {
            int nalSize = ((packet[offset] & 0xFF) << 8) | (packet[offset + 1] & 0xFF);
            offset += 2;
            if (nalSize <= 0 || offset + nalSize > length) {
                return;
            }
            queueAccessUnit(packet, offset, nalSize);
            offset += nalSize;
        }
    }

    private void processFuA(byte[] packet, int offset, int length) throws Exception {
        if (offset + 2 >= length) {
            return;
        }
        int indicator = packet[offset] & 0xFF;
        int header = packet[offset + 1] & 0xFF;
        boolean start = (header & 0x80) != 0;
        boolean end = (header & 0x40) != 0;
        int reconstructed = (indicator & 0xE0) | (header & 0x1F);
        int payloadOffset = offset + 2;
        int payloadLength = length - payloadOffset;

        if (start) {
            fuBuffer = new byte[262144];
            fuLength = 0;
            appendFuByte((byte) reconstructed);
        }
        if (fuBuffer == null) {
            return;
        }
        ensureFuCapacity(fuLength + payloadLength);
        System.arraycopy(packet, payloadOffset, fuBuffer, fuLength, payloadLength);
        fuLength += payloadLength;
        if (end) {
            queueAccessUnit(fuBuffer, 0, fuLength);
            fuBuffer = null;
            fuLength = 0;
        }
    }

    private void appendFuByte(byte value) {
        ensureFuCapacity(fuLength + 1);
        fuBuffer[fuLength++] = value;
    }

    private void ensureFuCapacity(int required) {
        if (fuBuffer.length >= required) {
            return;
        }
        fuBuffer = Arrays.copyOf(fuBuffer, Math.max(required, fuBuffer.length * 2));
    }

    private void queueAccessUnit(byte[] nal, int offset, int length) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream(length + START_CODE.length);
        out.write(START_CODE);
        out.write(nal, offset, length);
        byte[] data = out.toByteArray();

        int input = decoder.dequeueInputBuffer(10000);
        if (input < 0) {
            return;
        }
        ByteBuffer inputBuffer = decoder.getInputBuffer(input);
        if (inputBuffer == null) {
            decoder.queueInputBuffer(input, 0, 0, System.nanoTime() / 1000L, 0);
            return;
        }
        inputBuffer.clear();
        if (data.length > inputBuffer.remaining()) {
            lastError = "decoder input too small for NAL " + data.length;
            decoder.queueInputBuffer(input, 0, 0, System.nanoTime() / 1000L, 0);
            return;
        }
        inputBuffer.put(data);
        decoder.queueInputBuffer(input, 0, data.length, System.nanoTime() / 1000L, 0);
    }

    private byte[] yuv420ToRgba(Image image) {
        if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
            lastError = "unsupported decoder image format";
            return null;
        }
        Rect crop = image.getCropRect();
        int width = crop != null ? crop.width() : image.getWidth();
        int height = crop != null ? crop.height() : image.getHeight();
        int cropLeft = crop != null ? crop.left : 0;
        int cropTop = crop != null ? crop.top : 0;
        if (width <= 0 || height <= 0) {
            lastError = "empty decoder image";
            return null;
        }
        frameWidth = width;
        frameHeight = height;
        byte[] out = new byte[width * height * 4];
        Image.Plane[] planes = image.getPlanes();
        if (planes == null || planes.length < 3) {
            lastError = "decoder image missing YUV planes";
            return null;
        }
        ByteBuffer yPlane = planes[0].getBuffer();
        ByteBuffer uPlane = planes[1].getBuffer();
        ByteBuffer vPlane = planes[2].getBuffer();
        if (yPlane == null || uPlane == null || vPlane == null) {
            lastError = "decoder image has null YUV plane";
            return null;
        }
        yPlane = yPlane.duplicate();
        uPlane = uPlane.duplicate();
        vPlane = vPlane.duplicate();
        int yRowStride = planes[0].getRowStride();
        int yPixelStride = Math.max(1, planes[0].getPixelStride());
        int uRowStride = planes[1].getRowStride();
        int vRowStride = planes[2].getRowStride();
        int uPixelStride = Math.max(1, planes[1].getPixelStride());
        int vPixelStride = Math.max(1, planes[2].getPixelStride());

        int outIndex = 0;
        for (int y = 0; y < height; y++) {
            int sourceY = cropTop + y;
            int uvY = sourceY / 2;
            for (int x = 0; x < width; x++) {
                int sourceX = cropLeft + x;
                int uvX = sourceX / 2;
                int yIndex = sourceY * yRowStride + sourceX * yPixelStride;
                int uIndex = uvY * uRowStride + uvX * uPixelStride;
                int vIndex = uvY * vRowStride + uvX * vPixelStride;
                if (yIndex < 0 || yIndex >= yPlane.limit()
                        || uIndex < 0 || uIndex >= uPlane.limit()
                        || vIndex < 0 || vIndex >= vPlane.limit()) {
                    lastError = "decoder image plane bounds";
                    return null;
                }
                int yValue = yPlane.get(yIndex) & 0xFF;
                int uValue = uPlane.get(uIndex) & 0xFF;
                int vValue = vPlane.get(vIndex) & 0xFF;
                int c = yValue - 16;
                int d = uValue - 128;
                int e = vValue - 128;
                int r = clamp((298 * c + 409 * e + 128) >> 8);
                int g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
                int b = clamp((298 * c + 516 * d + 128) >> 8);
                out[outIndex++] = (byte) r;
                out[outIndex++] = (byte) g;
                out[outIndex++] = (byte) b;
                out[outIndex++] = (byte) 255;
            }
        }
        return out;
    }

    private static int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }
}
