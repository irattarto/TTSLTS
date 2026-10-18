# TTSLTS

An offline system tray app for Windows with two features:

- **Hungarian text to speech** with the [Piper](https://github.com/rhasspy/piper) voices Anna, Berta and Imre, or with [F5-TTS](https://github.com/SWivid/F5-TTS) voices of your own.
- **Real-time voice changing** with [RVC v2](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) voice models.
- **Training your own voice** from mp3 or wav recordings, saved as an RVC v2 `.pth` and an `.onnx` file (PyTorch is set up on first use).

Everything runs locally through ONNX Runtime (training uses PyTorch). Nothing is sent over the network except the one-time model downloads.

## Install

Requires Python 3.10–3.13. A GPU is needed for the real-time voice changer (a CPU is about four times too slow); text to speech runs fine on CPU. The Windows installer and portable build use the GPU through DirectML, which works with NVIDIA, AMD and Intel graphics cards.

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -e .

:: NVIDIA GPU (CUDA 12): swap in the CUDA build of ONNX Runtime
pip install -e .[gpu]
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu

:: or any Windows GPU (NVIDIA, AMD, Intel) with DirectML instead of CUDA
pip uninstall -y onnxruntime
pip install onnxruntime-directml
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

1. Settings > Models > **Download Hungarian voices** (about 190 MB, from the original [Piper voice collection](https://huggingface.co/rhasspy/piper-voices); you are asked to accept the terms first, and **Speak** offers it too while no voice is installed). The voices Anna, Berta and Imre use CC0 training data but were fine-tuned from Piper's US English *lessac* voice, whose data is licensed for research use, so keep them to research and personal use.
2. On the **Text to Speech** tab, type text, pick a voice and speed, then **Speak** or **Save WAV…**. Anna, Berta and Imre are three different speakers, so listen to each with your own text and keep the one you like. Voice variation (default 0.67) and Rhythm variation (default 0.80) make the speech steadier when lowered and livelier when raised; well above the defaults it gets shaky. Common Hungarian abbreviations (kb., pl., dr., stb.) and units are read as words, and pauses follow the punctuation (longer after a question, "…", a line break or a new paragraph).
3. **Play on** picks the output device. Choose a virtual cable to send speech into Discord, games or OBS.
   **Add voice…** (next to the voice list) adds a Piper voice of your own: choose its `.onnx` file, with the `.onnx.json` settings file of the same name next to it. It is copied to `%APPDATA%\TTSLTS\models\piper\custom` and listed as *name (own Piper voice)*; the downloaded ones as *Anna (Piper)* and so on, so the list always shows which engine reads.
   **Pronunciations…** (button on the Text to Speech tab, or select a word in the text and right-click it) lists words the voice says wrongly. Write each one the way it sounds in the language of the text, using that language's spelling: `Windows` -> `vindóz`, `Nvidia` -> `envídia`. Whole words only (`Windows-ban` works); *Match case* limits an entry to the exact capitalisation; **Listen** speaks the second column so you can check it. The file is `%APPDATA%\TTSLTS\pronunciations.json`. For Hungarian speech a built-in list also says FBI, CIA, USA, USB, DVD, PC, AI, IT and a few more the way Hungarians do, with English letter names (switch it off in the same window; an entry of your own wins).
4. For speech in a voice of your own, use an own Piper voice or an own F5-TTS voice (**Add voice…**, see below). Running the speech through an RVC voice was removed: measured, it made more mistakes (10.7-12.5 % of the letters wrong for Whisper) than an own Piper voice trained on the same narrator (9.0 %).

#### F5-TTS voices

An F5-TTS voice reads more smoothly and clearly than Piper (measured on one narrator: 2.5 % of the letters wrong for Whisper against 6.8 %, and less pitch trembling), but it needs PyTorch and a graphics card, and takes a few seconds per sentence. No F5 voice is shipped or downloaded: you add your own (the Hungarian F5 base model, [Maxdorger29/f5-tts-hungarian](https://huggingface.co/Maxdorger29/f5-tts-hungarian), is CC BY-NC 4.0, non-commercial).

1. Settings > Models > **Set up F5-TTS voices** (asks first): a private environment in `%APPDATA%\TTSLTS\runtime-f5` with Python, PyTorch, F5-TTS and faster-whisper, about 4 GB with an NVIDIA GPU. The first reading also fetches the Vocos vocoder and the Whisper *small* model from Hugging Face. Delete the folder to remove it.
2. **Add voice…** and choose the voice's `.safetensors` file (F5TTS_v1_Base weights). Next to it must be `vocab.txt` (its vocabulary), `reference.wav` (a calm 5-15 s sample of the voice) and `reference.txt` (the sample's exact text). It is copied to `%APPDATA%\TTSLTS\models\f5\<folder name>` and listed as *name (own F5 voice)*.
3. Speak as usual: the first sentence plays after a few seconds while the next ones are being made. Voice and Rhythm variation are Piper settings and are off for F5 voices (an F5 voice takes its manner from its sample). The voice is loaded in its own process the first time (up to a minute) and stays loaded; it moves off the graphics card while the voice changer runs (it is then slow) and is closed when the voice changer or training starts. Its log is `%APPDATA%\TTSLTS\f5.log`.

Three things are done for the Hungarian F5 model: numbers are written out as Hungarian words first (it saw almost no digits in training and read them wrongly; dates, times, ordinals, decimals, percentages, money and number + suffix such as *1848. március 15-én*, *10:30-kor*, *3,5%-kal*, *5-öt* are covered), the text is lowercased (its vocabulary has no capitals, a capital is read as a space), and every sentence starts with the filler word "nos," that is cut off again where Whisper hears the first real word. Without it F5 often puts a stray sound in front ("Ugalambos"); a sentence whose start still comes out wrong is read once more. Abbreviations in capitals are said letter by letter (*SNS* as *es-en-es*; *NATO*, *MÁV* stay words). How lively a sentence sounds depends on F5's random start: measured, about four sentences in ten came out with a flat melody, which sounds tired, so a clearly flat one (pitch range under 5 semitones) is read once more and the livelier reading is kept. After the filler is cut off, Whisper listens to the new start once more, and a reading that does not begin with the text's first word is read again (Whisper's word times are sometimes off by most of a second).

### Voice changer

You supply the models (they are never bundled or committed):

| Field | File | Notes |
|---|---|---|
| RVC v2 voice | a downloaded `.pth` voice, or an `.onnx` export | The `.pth` file that voices are shared as works directly: the first use converts it to ONNX on your computer (a few seconds, cached). Only **v2** voices with pitch work (the title or model card says v2; 768-dim). Older `.onnx` exports from RVC WebUI / w-okada also work, in both input layouts and fp16. |
| ContentVec | `vec-768-layer-12.onnx` (or another 768-dim ContentVec/HuBERT ONNX) | RVC v2 models use 768-dim features. |
| RMVPE (optional) | `rmvpe.onnx` from the w-okada voice changer | Better pitch tracking. Without it, a built-in YIN pitch tracker is used. |
| RVC sample rate | 32/40/48 kHz | Only used when the ONNX file doesn't record its own rate (w-okada exports do). |

Then on the **Voice Changer** tab pick your microphone and output and press **Start**. To use the changed voice in other apps, set Output to a virtual cable such as [VB-Audio CABLE](https://vb-audio.com/Cable/) (*CABLE Input*) and select *CABLE Output* as the microphone in the other app.

**Index rate** and **Protect consonants** (Voice Changer tab) make a voice trained in TTSLTS clearer and closer to the real recordings: training saves the features of your recordings next to the voice (`<name>.index.npy`, made on first use for older voices), and each frame of the converted speech is mixed with its closest real frames. Index rate 0-0.3 keeps the pronunciation clearest (default 0.2; measured, higher values blur Hungarian sounds such as r/l and sz/s, 0 switches it off); Protect keeps s, sh, f and breath sounds clear. Voices from elsewhere have no index, so these two do nothing for them; both can be changed while the voice changer runs. Joins between chunks are lined up with SOLA and crossfaded band by band, so the hiss of "s"/"sz" keeps its level through every join. The silence gate keeps quiet word endings and stays open 200 ms after the last sound.

Latency is roughly chunk size + crossfade + processing time + a 40 ms safety margin (it grows by itself if the computer is sometimes slower) + the output device's own delay. Lower **Chunk size** for less delay, raise it if the voice crackles. **Context** is how much earlier audio the model sees each time: 1500 ms (the default) converts clearly better than 600 ms at no extra delay, only more GPU work. The **Processing** readout should stay well under the chunk size. A WASAPI output adds the least delay (about 20 ms against 90 ms for MME, measured with a virtual cable).

### Train your own voice

The **Train** tab makes an RVC v2 voice from recordings of one speaker, in the installed app as well as from source. PyTorch is too big to ship in the installers (the CUDA build is several GB), so the first time you press **Start training** the app asks, then sets up its own private environment in `%APPDATA%\TTSLTS\runtime` (Python from python.org, PyTorch and a few packages; about 3 GB with an NVIDIA GPU, 0.5 GB without; training on a CPU takes many hours). Delete that folder to remove it. Running from source with PyTorch already installed (`pip install -e .[train]`) uses that instead and downloads nothing.

1. Settings > Models > **Download ContentVec + RMVPE** (once, also needed by the voice changer).
2. On the **Train** tab add your recordings (mp3, wav, flac, ogg). Use clean speech of one person without music or noise; about 10 minutes gives a good voice and 2-3 minutes already works.
3. Give the voice a name, pick the sample rate (40 kHz is the usual choice), epochs (100-300 for a few minutes of audio) and batch size (lower it if the GPU runs out of memory), then **Start training**. The first time you are asked to accept the terms and download the pretrained RVC v2 models (about 215 MB) that training starts from.
4. Training cuts the recordings at pauses, extracts ContentVec features and pitch, and fine-tunes the pretrained model. The voice file is refreshed every *Save every* epochs, so you can try it before training ends. **Stop** saves and ends early; starting the same name again offers to continue where it stopped.
5. The result is saved to `%APPDATA%\TTSLTS\voices\<name>\` as `<name>.pth` (the format voices are shared in, works in other RVC tools) and `<name>.onnx`, plus `<name>.index.npy` (the voice index) and `<name>.json` (the voice's natural pitch). **Use this voice** selects it for the voice changer.

Training runs in a separate process (`python -m ttslts.train.run`, from the app's bundled sources in the installed app); its log is `%APPDATA%\TTSLTS\train\<name>\train.log`. Only train on voices you have permission to use.

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
  tts/              Piper engine, Hungarian voices download, text clean-up, pronunciations;
                    F5-TTS voices (f5_engine.py drives f5_worker.py in the F5 environment)
  vc/rvc.py         RVC v2 ONNX inference (ContentVec + pitch + voice model)
  vc/rvc_pth.py     .pth voice checkpoint -> ONNX importer (no PyTorch needed)
  vc/realtime.py    chunked streaming conversion with crossfades
  train/            voice training: prepare.py (cut clips, features), trainer.py (loop), run.py (job process)
  train/rvc_model/  original RVC network code (MIT, see its LICENSE), used only for training
  ui/               main window, tray icon and menu
```

## Licenses and consent

RVC is MIT, but check the license of each voice model you use; the Hungarian Piper voices are for research and personal use (see above), and F5 voices made from the Hungarian F5 model are non-commercial (CC BY-NC 4.0). The pretrained models used for training carry the RVC author's research-use clause (shown when you download them), and `train/rvc_model/` is RVC's MIT-licensed model code. Only clone voices you have permission to use.

## Android app (text to speech only)

`android/` contains a native Android app that speaks Hungarian offline with the Piper voices through [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). It does text to speech only; the RVC voice changer is desktop-only because real-time voice conversion is too heavy for phones.

- Anna, Berta and Imre download on demand (about 65 MB each) after you accept their terms (research and personal use), then everything works offline. The Kokoro model of older versions is deleted on the first start to free the space.
- Pick a voice and speed, then **Speak**, or **Save WAV**. Sharing text to TTSLTS from another app also fills the text box.
- Download `TTSLTS-android.apk` from the latest [release](../../releases) and allow installing from unknown sources. Release APKs are debug-signed, so installing a newer build over an older one may require uninstalling first.

Build it yourself (JDK 17, Android SDK, Gradle 8.9+):

```sh
./android/fetch-native.sh          # sherpa-onnx native libraries
cd android && gradle assembleDebug # APK in app/build/outputs/apk/debug/
```

## Releases

Pushing a tag such as `v0.1.0` (or running the Release workflow by hand from the Actions tab with a tag name, which creates the tag) builds these downloads and attaches them to a GitHub pre-release:

| File | For |
|---|---|
| `TTSLTS-Setup.exe` | Windows installer (per-user, no administrator needed) |
| `TTSLTS-windows.zip` | Windows portable folder, run `TTSLTS.exe` |
| `ttslts_<version>_amd64.deb` | Linux installer for Debian, Ubuntu and derivatives (`sudo apt install ./ttslts_*.deb`) |
| `TTSLTS-linux-x86_64.tar.gz` | Linux portable folder, run `TTSLTS` |
| `TTSLTS-android.apk` | Android, text to speech only |

On Linux the tray icon needs a desktop that shows tray icons (GNOME needs an extension).
