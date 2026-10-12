# TTSLTS

An offline system tray app for Windows with two features:

- **Text to speech** with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0), 54 preset voices in 9 languages.
- **Real-time voice changing** with [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) voice models.

Everything runs locally through ONNX Runtime. Nothing is sent over the network except the one-time Kokoro model download.

## Install

Requires Python 3.10–3.13. An NVIDIA GPU is recommended for the voice changer; text to speech runs fine on CPU.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .

:: NVIDIA GPU (CUDA 12): swap in the CUDA build of ONNX Runtime
pip install -e .[gpu]
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu
```

Run it with `ttslts` (or `pythonw -m ttslts` to start without a console window).

## Using it

The app lives in the system tray. Click the tray icon to show or hide the window; right-click for **Voice changer** on/off, **Speak clipboard**, **Stop speaking** and **Quit**.

**Settings > Window**

| Option | Default | What it does |
|---|---|---|
| Minimize to tray | on | Minimizing hides the window to the tray instead of the taskbar |
| Close to tray | on | The window's close button hides it to the tray; use **Quit** in the tray menu to exit |
| Start minimized to tray | off | Start with only the tray icon showing |

Settings are saved automatically to `%APPDATA%\TTSLTS\settings.json`.

### Text to speech

1. Settings > Models > **Download Kokoro** (about 340 MB, saved to `%APPDATA%\TTSLTS\models\kokoro`). You can also point the two fields at your own `kokoro-v1.0.onnx` and `voices-v1.0.bin`.
2. On the **Text to Speech** tab, type text, pick a voice, language and speed, then **Speak** or **Save WAV…**.
3. **Play on** picks the output device. Choose a virtual cable to send speech into Discord, games or OBS.
4. Tick **Speak with my voice model (RVC)** to run Kokoro's output through your RVC voice, giving TTS your custom voice.

### Voice changer

You supply the models (they are never bundled or committed):

| Field | File | Notes |
|---|---|---|
| RVC v2 voice | your voice model exported to ONNX | Export from RVC WebUI (its *Export Onnx* tab) or the w-okada voice changer. Both input layouts are supported, as are fp16 exports. |
| ContentVec | `vec-768-layer-12.onnx` (or another 768-dim ContentVec/HuBERT ONNX) | RVC v2 models use 768-dim features. |
| RMVPE (optional) | `rmvpe.onnx` from the w-okada voice changer | Better pitch tracking. Without it, a built-in YIN pitch tracker is used. |
| RVC sample rate | 32/40/48 kHz | Only used when the ONNX file doesn't record its own rate (w-okada exports do). |

Then on the **Voice Changer** tab pick your microphone and output and press **Start**. To use the changed voice in other apps, set Output to a virtual cable such as [VB-Audio CABLE](https://vb-audio.com/Cable/) (*CABLE Input*) and select *CABLE Output* as the microphone in the other app.

Latency is roughly chunk size + crossfade + processing time. Lower **Chunk size** for less delay, raise it if the voice crackles. **Context** is how much earlier audio the model sees each time (more is smoother but slower). The **Processing** readout should stay well under the chunk size.

## Development

```bash
pip install -e .[dev]
ruff check . && ruff format --check .
pytest
```

Layout:

```
src/ttslts/
  app.py            entry point (QApplication + tray)
  controller.py     background work for TTS, voice changer, downloads
  settings.py       JSON settings (tray options default on)
  audio/            DSP helpers (resampling, YIN pitch, crossfade), devices, WAV
  tts/              Kokoro engine and model download
  vc/rvc.py         RVC v2 ONNX inference (ContentVec + pitch + voice model)
  vc/realtime.py    chunked streaming conversion with crossfades
  ui/               main window, tray icon and menu
```

## Licenses and consent

Kokoro is Apache-2.0 and RVC is MIT, so both are usable commercially, but check the license of each voice model you use. Only clone voices you have permission to use.

## Android app (text to speech only)

`android/` contains a native Android app that runs the same Kokoro voices offline through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). It does text to speech only; the RVC voice changer is desktop-only because real-time voice conversion is too heavy for phones.

- On first launch the app downloads the Kokoro model once (several hundred MB), then works offline.
- Pick a voice and speed, then **Speak**, or **Save WAV**. Sharing text to TTSLTS from another app also fills the text box.
- Download `TTSLTS-android.apk` from the latest [release](../../releases) and allow installing from unknown sources. Release APKs are debug-signed, so installing a newer build over an older one may require uninstalling first.

Build it yourself (JDK 17, Android SDK, Gradle 8.9+):

```sh
./android/fetch-native.sh          # sherpa-onnx native libraries
cd android && gradle assembleDebug # APK in app/build/outputs/apk/debug/
```

## Releases

Pushing a tag such as `v0.1.0` runs the Release workflow, which builds these downloads and attaches them to a GitHub pre-release:

| File | For |
|---|---|
| `TTSLTS-Setup.exe` | Windows installer (per-user, no administrator needed) |
| `TTSLTS-windows.zip` | Windows portable folder, run `TTSLTS.exe` |
| `ttslts_<version>_amd64.deb` | Linux installer for Debian, Ubuntu and derivatives (`sudo apt install ./ttslts_*.deb`) |
| `TTSLTS-linux-x86_64.tar.gz` | Linux portable folder, run `TTSLTS` |
| `TTSLTS-android.apk` | Android, text to speech only |

On Linux the tray icon needs a desktop that shows tray icons (GNOME needs an extension).
