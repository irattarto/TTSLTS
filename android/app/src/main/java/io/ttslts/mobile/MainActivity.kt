package io.ttslts.mobile

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.util.concurrent.Executors

/** Single-screen offline Hungarian text to speech with the Piper voices. */
class MainActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var speaker: Speaker

    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var download: Button
    private lateinit var input: EditText
    private lateinit var voice: Spinner
    private lateinit var speedLabel: TextView
    private lateinit var speed: SeekBar
    private lateinit var speak: Button
    private lateinit var stop: Button
    private lateinit var save: Button

    private var busy = false
    private var ready = false
    private var piperIndex = 0

    private val saveWav =
        registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
            if (uri != null) exportWav(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        speaker = Speaker(applicationContext)
        // start on a voice that is already downloaded, if there is one
        piperIndex = PiperVoices.all.indexOfFirst { PiperVoices.isInstalled(this, it) }.coerceAtLeast(0)
        setContentView(buildUi())

        if (savedInstanceState == null && intent?.action == Intent.ACTION_SEND) {
            intent.getStringExtra(Intent.EXTRA_TEXT)?.let { input.setText(it) }
        }

        // The Kokoro model of older versions (several hundred MB) is no longer used.
        worker.execute { ModelStore.removeOldModels(applicationContext) }
        val info = currentVoice()
        if (PiperVoices.isInstalled(this, info)) loadPiper(info) else showNeedsVoice(info)
    }

    override fun onDestroy() {
        speaker.stop()
        worker.execute { speaker.release() }
        worker.shutdown()
        super.onDestroy()
    }

    // --- UI -------------------------------------------------------------------------------

    private fun buildUi(): View {
        val pad = dp(16)
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        column.addView(TextView(this).apply {
            text = "TTSLTS"
            textSize = 24f
        })
        status = TextView(this).apply { setPadding(0, dp(8), 0, dp(8)) }
        column.addView(status)

        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            visibility = View.GONE
        }
        column.addView(progress, matchWidth())

        download = Button(this).apply {
            text = "Download Hungarian voice"
            setOnClickListener { ensurePiperAndLoad() }
        }
        column.addView(download, matchWidth())

        input = EditText(this).apply {
            hint = "Type or paste text to speak"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            minLines = 5
            gravity = Gravity.TOP or Gravity.START
        }
        column.addView(input, matchWidth().apply { topMargin = dp(12) })

        column.addView(TextView(this).apply {
            text = "Voice"
            setPadding(0, dp(12), 0, 0)
        })
        voice = Spinner(this).apply {
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                    // Only a real change of the voice reloads; filling the list does not.
                    if (pos != piperIndex) {
                        piperIndex = pos
                        if (!busy) ensurePiperAndLoad()
                    }
                }

                override fun onNothingSelected(parent: AdapterView<*>?) = Unit
            }
        }
        voice.tooltipText = PiperVoices.HELP_VOICE
        column.addView(voice, matchWidth())

        column.addView(TextView(this).apply {
            text = PiperVoices.HINT
            setPadding(0, dp(4), 0, 0)
        })

        speedLabel = TextView(this).apply { setPadding(0, dp(12), 0, 0) }
        column.addView(speedLabel)
        speed = SeekBar(this).apply {
            max = 150 // 0.5x .. 2.0x
            progress = 50 // 1.0x
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(bar: SeekBar?, value: Int, fromUser: Boolean) {
                    updateSpeedLabel()
                }

                override fun onStartTrackingTouch(bar: SeekBar?) = Unit
                override fun onStopTrackingTouch(bar: SeekBar?) = Unit
            })
        }
        speed.tooltipText = PiperVoices.HELP_SPEED
        column.addView(speed, matchWidth())
        updateSpeedLabel()

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        speak = Button(this).apply {
            text = "Speak"
            setOnClickListener { onSpeak() }
        }
        stop = Button(this).apply {
            text = "Stop"
            setOnClickListener { speaker.stop() }
        }
        save = Button(this).apply {
            text = "Save WAV"
            setOnClickListener { onSave() }
        }
        for (b in listOf(speak, stop, save)) {
            buttons.addView(b, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        column.addView(buttons, matchWidth().apply { topMargin = dp(12) })

        populatePiperVoices()
        refreshControls()
        return ScrollView(this).apply { addView(column) }
    }

    private fun refreshControls() {
        download.visibility = if (!ready && !busy) View.VISIBLE else View.GONE
        val enabled = ready && !busy
        speak.isEnabled = enabled
        save.isEnabled = enabled
        voice.isEnabled = !busy
        stop.isEnabled = ready
    }

    private fun updateSpeedLabel() {
        speedLabel.text = "Speed: %.2fx".format(currentSpeed())
    }

    private fun currentSpeed() = 0.5f + speed.progress / 100f

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    private fun matchWidth() =
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)

    // --- Voices ---------------------------------------------------------------------------

    private fun currentVoice() = PiperVoices.all[piperIndex.coerceIn(0, PiperVoices.all.lastIndex)]

    private fun populatePiperVoices() {
        val labels = PiperVoices.all.map { v ->
            val label = "${v.name} - ${v.note}"
            if (PiperVoices.isInstalled(this, v)) label else "$label (not downloaded)"
        }
        voice.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        voice.setSelection(piperIndex.coerceIn(0, labels.lastIndex))
    }

    private fun showNeedsVoice(info: PiperVoiceInfo) {
        ready = false
        status.text = "The Hungarian voice ${info.name} is not downloaded yet (about 65 MB). " +
            "Everything works offline afterwards."
        refreshControls()
    }

    private fun ensurePiperAndLoad() {
        val info = currentVoice()
        if (PiperVoices.isInstalled(this, info)) loadPiper(info) else askPiperConsent(info)
    }

    private fun askPiperConsent(info: PiperVoiceInfo) {
        showNeedsVoice(info)
        AlertDialog.Builder(this)
            .setTitle(PiperVoices.CONSENT_TITLE)
            .setMessage(PiperVoices.CONSENT_TEXT)
            .setPositiveButton("I accept, download") { _, _ -> downloadPiper(info) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun downloadPiper(info: PiperVoiceInfo) {
        busy = true
        status.text = "Downloading ${info.name}…"
        progress.visibility = View.VISIBLE
        progress.isIndeterminate = true
        refreshControls()
        worker.execute {
            try {
                ModelStore.downloadArchive(this, info.dirName, info.modelFile) { done, total ->
                    runOnUiThread {
                        if (total > 0) {
                            progress.isIndeterminate = false
                            progress.max = 1000
                            progress.progress = (done * 1000 / total).toInt()
                            status.text = "Downloading ${info.name}… %d / %d MB"
                                .format(done shr 20, total shr 20)
                        } else {
                            status.text = "Downloading ${info.name}… ${done shr 20} MB"
                        }
                    }
                }
                runOnUiThread {
                    progress.visibility = View.GONE
                    busy = false
                    populatePiperVoices()
                    loadPiper(info)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    progress.visibility = View.GONE
                    busy = false
                    ready = false
                    status.text = "Download failed: ${e.message}"
                    refreshControls()
                }
            }
        }
    }

    private fun loadPiper(info: PiperVoiceInfo) {
        busy = true
        status.text = "Loading ${info.name}…"
        refreshControls()
        worker.execute {
            try {
                speaker.loadPiper(info)
                runOnUiThread {
                    ready = true
                    busy = false
                    status.text = "Ready. Running fully offline."
                    refreshControls()
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    ready = false
                    busy = false
                    status.text = "Could not load the voice: ${e.message}"
                    refreshControls()
                }
            }
        }
    }

    // --- Actions --------------------------------------------------------------------------

    private fun onSpeak() {
        val text = input.text.toString().trim()
        if (text.isEmpty()) return toast("Enter some text first")
        val speedValue = currentSpeed()
        busy = true
        status.text = "Speaking…"
        refreshControls()
        stop.isEnabled = true
        worker.execute {
            val error = runCatching { speaker.speak(text, 0, speedValue) }.exceptionOrNull()
            runOnUiThread {
                busy = false
                status.text = error?.let { "Speech failed: ${it.message}" } ?: "Ready."
                refreshControls()
            }
        }
    }

    private fun onSave() {
        if (input.text.toString().isBlank()) return toast("Enter some text first")
        saveWav.launch("speech.wav")
    }

    private fun exportWav(uri: Uri) {
        val text = input.text.toString().trim()
        val speedValue = currentSpeed()
        busy = true
        status.text = "Saving WAV…"
        refreshControls()
        worker.execute {
            val error = runCatching {
                contentResolver.openOutputStream(uri)?.use { speaker.writeWav(text, 0, speedValue, it) }
                    ?: error("Could not open the file for writing")
            }.exceptionOrNull()
            runOnUiThread {
                busy = false
                status.text = error?.let { "Save failed: ${it.message}" } ?: "Saved."
                refreshControls()
            }
        }
    }

    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
}
