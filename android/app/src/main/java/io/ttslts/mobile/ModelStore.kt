package io.ttslts.mobile

import android.content.Context
import java.io.File
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream

/** Downloads and locates the Kokoro model files (sherpa-onnx packaging, 53 voices). */
object ModelStore {
    private const val MODEL_DIR = "kokoro-multi-lang-v1_0"
    private const val URL =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$MODEL_DIR.tar.bz2"

    fun modelDir(context: Context): File = File(context.filesDir, "models/$MODEL_DIR")

    fun isInstalled(context: Context): Boolean {
        val dir = modelDir(context)
        return File(dir, "model.onnx").isFile &&
            File(dir, "voices.bin").isFile &&
            File(dir, "tokens.txt").isFile &&
            File(dir, "espeak-ng-data").isDirectory
    }

    /**
     * Downloads the archive and unpacks it while streaming, so no temporary copy of the
     * archive is stored. Blocking; call from a background thread. [onProgress] gets
     * (downloaded bytes, total bytes or -1 when unknown).
     */
    fun download(context: Context, onProgress: (Long, Long) -> Unit) {
        val root = File(context.filesDir, "models")
        val staging = File(root, ".staging")
        staging.deleteRecursively()
        staging.mkdirs()

        val connection = (URL(URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 30_000
            instanceFollowRedirects = true
        }
        try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw IOException("Download failed: HTTP ${connection.responseCode}")
            }
            val total = connection.contentLengthLong
            val counted = CountingInputStream(connection.inputStream) { onProgress(it, total) }
            TarArchiveInputStream(BZip2CompressorInputStream(counted.buffered(1 shl 16))).use { tar ->
                val stagingPath = staging.canonicalPath + File.separator
                while (true) {
                    val entry = tar.nextEntry ?: break
                    val target = File(staging, entry.name)
                    // Refuse entries that would be written outside the staging directory.
                    if (!target.canonicalPath.startsWith(stagingPath)) {
                        throw IOException("Unsafe path in archive: ${entry.name}")
                    }
                    if (entry.isDirectory) {
                        target.mkdirs()
                    } else {
                        target.parentFile?.mkdirs()
                        target.outputStream().use { tar.copyTo(it) }
                    }
                }
            }
        } catch (e: Exception) {
            staging.deleteRecursively()
            throw e
        } finally {
            connection.disconnect()
        }

        val extracted = File(staging, MODEL_DIR)
        if (!File(extracted, "model.onnx").isFile) {
            staging.deleteRecursively()
            throw IOException("The downloaded archive does not contain $MODEL_DIR/model.onnx")
        }
        val dest = modelDir(context)
        dest.deleteRecursively()
        if (!extracted.renameTo(dest)) {
            staging.deleteRecursively()
            throw IOException("Could not move the model into place")
        }
        staging.deleteRecursively()
    }

    private class CountingInputStream(
        stream: InputStream,
        private val onBytes: (Long) -> Unit,
    ) : FilterInputStream(stream) {
        private var count = 0L
        private var lastReport = 0L

        override fun read(): Int = super.read().also { if (it >= 0) add(1) }

        override fun read(b: ByteArray, off: Int, len: Int): Int =
            super.read(b, off, len).also { if (it > 0) add(it.toLong()) }

        private fun add(n: Long) {
            count += n
            // Report about once per 256 KiB to keep UI updates cheap.
            if (count - lastReport >= 256 * 1024) {
                lastReport = count
                onBytes(count)
            }
        }
    }
}
