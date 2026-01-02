
#### ALSA related installing

```text
sudo apt update

sudo apt install -y \
    alsa-base \
    alsa-utils \
    alsa-tools \
    libasound2 \
    libasound2-dev \
    pulseaudio \
    pavucontrol
```

#### install Gstreamer with ALSA backends

```text

sudo apt update

sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-alsa \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-x \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav
```

#### kernel level checking

```text
ls -l /dev/snd/
```

#### ALSA level checking

```text
aplay -l
arecord -l
```

```text
soo@soo:~/Desktop$ aplay -l
**** List of PLAYBACK Hardware Devices ****
card 0: NVidia [HDA NVidia], device 3: HDMI 0 [HDMI 0]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 7: HDMI 1 [HDMI 1]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 8: HDMI 2 [HDMI 2]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 0: NVidia [HDA NVidia], device 9: HDMI 3 [HDMI 3]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 0: HDA Analog (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 1: HDA Digital (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 3: HDMI1 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 4: HDMI2 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 5: HDMI3 (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
soo@soo:~/Desktop$ arecord -l
**** List of CAPTURE Hardware Devices ****
card 1: sofhdadsp [sof-hda-dsp], device 0: HDA Analog (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 1: HDA Digital (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 6: DMIC (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 1: sofhdadsp [sof-hda-dsp], device 7: DMIC16kHz (*) []
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

#### Speaker Test

```text
speaker-test -D hw:0,0 -c 2 -t wav
```

in here Card is 0, and device is 0 for above speaker-test  

#### Microphone Test

##### mono test  
```text
arecord -D hw:0,0 -f S16_LE -r 16000 test.wav
aplay test.wav
```

##### stereo test
```text
arecord -D hw:0,0 -f S16_LE -c 2 -r 44100 test_stereo.wav
aplay test_stereo.wav
```

#### Microphone → Speaker (RAW PCM loopback)
```text
gst-launch-1.0 \
  alsasrc device=hw:0,0 ! \
  audio/x-raw,format=S16LE,rate=16000,channels=1 ! \
  audioconvert ! audioresample ! \
  alsasink device=hw:0,0
```

#### Microphone → WAV file
```text
gst-launch-1.0 \
  alsasrc device=hw:0,0 ! \
  audio/x-raw,format=S16LE,rate=16000,channels=1 ! \
  wavenc ! filesink location=mic.wav
```

#### Microphone → MP3
```text
gst-launch-1.0 \
  alsasrc device=hw:0,0 ! audioconvert ! audioresample ! \
  audio/x-raw,rate=44100,channels=2 ! \
  lamemp3enc target=quality quality=0 ! \
  filesink location=mic.mp3
```

#### WAV file → Speaker
```text
gst-launch-1.0 filesrc location=mic.wav ! wavparse ! audioconvert ! alsasink device=hw:0,0
```

#### MP3 file → Speaker
```text
gst-launch-1.0 filesrc location=mic.mp3 ! mpegaudioparse ! mpg123audiodec ! audioconvert ! alsasink device=hw:0,0
```

#### AAC file → Speaker
```text
gst-launch-1.0 filesrc location=mic.aac ! aacparse ! avdec_aac ! audioconvert ! alsasink device=hw:0,0
```

#### RTSP Audio (AAC / G711 / MP3)
```text
gst-launch-1.0 \
  rtspsrc location=rtsp://192.168.0.10:8554/audio latency=0 ! \
  decodebin ! audioconvert ! audioresample ! \
  alsasink device=hw:0,0
```

#### RAW PCM file → Speaker (no header)
```text
gst-launch-1.0 \
  filesrc location=audio.raw ! \
  audioparse format=s16le rate=16000 channels=1 ! \
  audioconvert ! alsasink device=hw:0,0
```