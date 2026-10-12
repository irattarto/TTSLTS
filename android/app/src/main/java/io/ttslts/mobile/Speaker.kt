package io.ttslts.mobile

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import java.io.File
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/** Wraps the sherpa-onnx Kokoro engine: streaming playback and WAV export. */
class Speaker(private val context: Context) {
    private var tts: OfflineTts? = null

    @Volatile
    private var stopRequested = false

    val numSpeakers: Int get() = tts?.numSpeakers() ?: 0

    /** Loads the model (a few seconds). Blocking; call from a background thread. */
    @Synchronized
    fun load() {
        if (tts != null) return
        val dir = ModelStore.modelDir(context)
        fun existing(vararg names: String) =
            names.map { File(dir, it) }.filter { it.exists() }.joinToString(",") { it.absolutePath }

        val kokoro = OfflineTtsKokoroModelConfig(
            model = File(dir, "model.onnx").absolutePath,
            voices = File(dir, "voices.bin").absolutePath,
            tokens = File(dir, "tokens.txt").absolutePath,
            dataDir = File(dir, "espeak-ng-data").absolutePath,
            lexicon = existing("lexicon-us-en.txt", "lexicon-zh.txt"),
            dictDir = File(dir, "dict").takeIf { it.isDirectory }?.absolutePath ?: "",
        )
        val config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                kokoro = kokoro,
                numThreads = max(2, Runtime.getRuntime().availableProcessors().coerceAtMost(4)),
                debug = false,
                provider = "cpu",
            ),
            ruleFsts = existing("phone-zh.fst", "date-zh.fst", "number-zh.fst"),
        )
        tts = OfflineTts(config = config)
    }

    fun stop() {
        stopRequested = true
    }

    /** Synthesizes and plays [text] as it is generated. Blocking; call off the main thread. */
    fun speak(text: String, speakerId: Int, speed: Float) {
        val engine = tts ?: error("Model is not loaded")
        stopRequested = false

        val sampleRate = engine.sampleRate()
        val minBuffer = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT,
        )
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setBufferSizeInBytes(max(minBuffer * 2, sampleRate * 4 / 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        var framesWritten = 0L
        try {
            track.play()
            engine.generateWithCallback(text, sid = speakerId, speed = speed) { samples ->
                track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
                framesWritten += samples.size
                if (stopRequested) 0 else 1
            }
            if (!stopRequested) {
                track.stop() // lets already-written audio play out
                val deadline = System.currentTimeMillis() + 60_000
                while (!stopRequested &&
                    track.playbackHeadPosition < framesWritten &&
                    System.currentTimeMillis() < deadline
                ) {
                    Thread.sleep(50)
                }
            }
        } finally {
            track.pause()
            track.flush()
            track.release()
        }
    }

    /** Synthesizes [text] and writes a 16-bit mono WAV to [out]. */
    fun writeWav(text: String, speakerId: Int, speed: Float, out: OutputStream) {
        val engine = tts ?: error("Model is not loaded")
        val audio = engine.generate(text, sid = speakerId, speed = speed)
        val pcm = ByteBuffer.allocate(audio.samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (s in audio.samples) {
            pcm.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray())
            putInt(36 + pcm.capacity())
            put("WAVEfmt ".toByteArray())
            putInt(16)
            putShort(1) // PCM
            putShort(1) // mono
            putInt(audio.sampleRate)
            putInt(audio.sampleRate * 2)
            putShort(2)
            putShort(16)
            put("data".toByteArray())
            putInt(pcm.capacity())
        }
        out.write(header.array())
        out.write(pcm.array())
        out.flush()
    }

    @Synchronized
    fun release() {
        tts?.release()
        tts = null
    }
}
