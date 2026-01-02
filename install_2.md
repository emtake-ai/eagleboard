#### Install V4L2 Core

```text
sudo apt update

sudo apt install -y \
    v4l-utils \
    libv4l-dev \
    libv4l-0 \
    qv4l2 \
    ffmpeg
```

#### Install gstreamer with related with V4L2


```text
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-video \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

#### check the which video device it has

```text
v4l2-ctl --list-devices
```

Input Type.

#### UVC Camera – RAW YUV (USB Webcam, Zero-Compression)

```text
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! \
  videoconvert ! autovideosink
```

#### UVC Camera – MJPEG (Compressed over USB)

```text
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  image/jpeg,width=1280,height=720,framerate=30/1 ! \
  jpegdec ! videoconvert ! autovideosink
```

#### MIPI Camera – RAW YUV (CSI-2)

```text
gst-launch-1.0 \
  v4l2src device=/dev/video0 io-mode=dmabuf ! \
  video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! \
  videoconvert ! autovideosink
```

#### File as Image (JPG / PNG)

```text
gst-launch-1.0 \
  filesrc location=frame.jpg ! jpegdec ! videoconvert ! autovideosink
```

#### File as RAW YUV

```text
gst-launch-1.0 \
  filesrc location=frame.yuv ! \
  videoparse format=nv12 width=640 height=480 framerate=30/1 ! \
  videoconvert ! autovideosink
```

#### File as MP4 (H.264 inside)

```text
gst-launch-1.0 \
  filesrc location=video.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```