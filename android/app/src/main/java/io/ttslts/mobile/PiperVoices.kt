package io.ttslts.mobile

import android.content.Context
import java.io.File

/** A Hungarian Piper voice as packaged by sherpa-onnx (`vits-piper-<id>`). */
data class PiperVoiceInfo(val id: String, val name: String, val note: String) {
    val dirName: String get() = "vits-piper-$id"
    val modelFile: String get() = "$id.onnx"
}

object PiperVoices {
    val all = listOf(
        PiperVoiceInfo("hu_HU-anna-medium", "Anna", "higher pitch, about 175 Hz"),
        PiperVoiceInfo("hu_HU-berta-medium", "Berta", "higher pitch, about 170 Hz"),
        PiperVoiceInfo("hu_HU-imre-medium", "Imre", "lower pitch, about 115 Hz"),
    )

    // Help texts (long-press a setting to see them) and a visible tip for the Piper engine.
    const val HELP_ENGINE =
        "Kokoro: 53 voices in nine languages; Hungarian there is experimental and accented. " +
            "Piper: real Hungarian voices (Anna, Berta, Imre), downloaded on demand."
    const val HELP_VOICE =
        "Three different Hungarian speakers. They differ in pitch and character, so listen to " +
            "each with your own text and keep the one you like."
    const val HELP_SPEED = "Speaking speed. 1.0x is normal; slightly slower (0.9x) can help clarity."
    const val HINT =
        "Tip: Piper has three real Hungarian voices (Anna, Berta, Imre). Try all of them with " +
            "your own text and keep the one you like. Long-press a setting for details."

    const val CONSENT_TITLE = "Hungarian voices (Piper): terms of use"
    const val CONSENT_TEXT =
        "The Hungarian voices Anna, Berta and Imre come from the Piper voice collection " +
            "(huggingface.co/rhasspy/piper-voices), packaged by sherpa-onnx.\n\n" +
            "Their model cards state the training data is CC0 (public domain). The voices were " +
            "then fine-tuned from Piper's US English \"lessac\" voice, and the training data of " +
            "that voice (the Blizzard 2013 data from Lessac Technologies) is licensed for " +
            "research use. The Piper project adds no license of its own and leaves it to users " +
            "to respect the original ones.\n\n" +
            "By continuing you accept this: use these voices for research and personal " +
            "experiments, and do not use them for commercial products.\n\n" +
            "Download size: about 65 MB per voice."

    fun isInstalled(context: Context, voice: PiperVoiceInfo): Boolean {
        val dir = ModelStore.modelDir(context, voice.dirName)
        return File(dir, voice.modelFile).isFile &&
            File(dir, "tokens.txt").isFile &&
            File(dir, "espeak-ng-data").isDirectory
    }
}
