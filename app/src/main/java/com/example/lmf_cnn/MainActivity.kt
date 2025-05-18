package com.example.lmf_cnn // Make sure this matches your package name

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import android.content.Context
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.os.Environment // For saving output video
import androidx.lifecycle.lifecycleScope // For coroutines
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt
import android.widget.AdapterView // For Spinner
import android.widget.ArrayAdapter // For Spinner
import android.widget.Spinner        // For Spinner
import org.pytorch.Device
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.Image // For converting YUV to RGB
import android.renderscript.RenderScript // For YUV to RGB conversion (optional, API level dependent)
import java.nio.ByteBuffer // For MediaCodec buffers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive

// Constants for video processing
private const val TARGET_LR_HEIGHT = 120
// Width will be calculated based on aspect ratio, but let's assume a common one for now
// Or better, calculate it from the source video.
// For the model, it expects 214 width for 120 height.
private const val TARGET_LR_WIDTH_FOR_MODEL = 214
private const val UPSCALE_FACTOR = 4 // From your model
private const val PREPROCESSED_FRAME_CHANNEL_CAPACITY = 10 // Adjust as needed
private const val INFERENCE_INPUT_CHANNEL_CAPACITY = 5  // Adjust as needed
private const val PROCESSED_BITMAP_CHANNEL_CAPACITY = 10 // Adjust as needed
private const val INFERENCE_BATCH_SIZE = 2 // Define batch size for inference

// Data class to hold a triplet of bitmaps for inference
data class FrameTriplet(val f1: Bitmap, val f2: Bitmap, val f3: Bitmap, val originalFrameIndex: Int)
// Data class to hold the output of inference
data class InferenceOutput(val hrBitmap: Bitmap, val originalFrameIndex: Int)

class MainActivity : AppCompatActivity() {

    private lateinit var buttonSelectVideo: Button
    private lateinit var textViewSelectedVideo: TextView
    private lateinit var buttonProcessVideo: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var textViewStatus: TextView
    private lateinit var spinnerDeviceSelection: Spinner // Declare Spinner


    private var selectedVideoUri: Uri? = null

    private var pytorchModule: Module? = null
    private val modelAssetName = "lmf_cnn_mobile.ptl"
    private var selectedDevice: Device = Device.CPU // Default to CPU
    private var currentModuleDevice: Device? = null // To track what device the current module is loaded on



    // ActivityResultLauncher for requesting permission
    private val requestPermissionLauncher: ActivityResultLauncher<String> =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.i("MainActivity", "Storage permission granted")
                openVideoPicker()
            } else {
                Log.e("MainActivity", "Storage permission denied")
                Toast.makeText(this, "Storage permission is required to select a video", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        buttonSelectVideo = findViewById(R.id.buttonSelectVideo)
        textViewSelectedVideo = findViewById(R.id.textViewSelectedVideo)
        buttonProcessVideo = findViewById(R.id.buttonProcessVideo)
        progressBar = findViewById(R.id.progressBar)
        textViewStatus = findViewById(R.id.textViewStatus)
        spinnerDeviceSelection = findViewById(R.id.spinnerDeviceSelection)

        setupDeviceSpinner() // <-- Add this line to initialize the spinner

        buttonSelectVideo.setOnClickListener {
            checkPermissionAndOpenVideoPicker()
        }

        buttonProcessVideo.setOnClickListener {
            selectedVideoUri?.let { uri ->
                if (pytorchModule == null) {
                    Toast.makeText(this, "Model not loaded yet!", Toast.LENGTH_LONG).show()
                    return@setOnClickListener
                }
                // Disable buttons and show progress
                buttonProcessVideo.isEnabled = false
                buttonSelectVideo.isEnabled = false
                progressBar.visibility = View.VISIBLE
                progressBar.isIndeterminate = false // We will set progress
                progressBar.progress = 0
                textViewStatus.text = "Starting processing on $selectedDevice..."

                // Launch coroutine for video processing
                lifecycleScope.launch {
                    processVideo(uri)
                }
            } ?: run {
                Toast.makeText(this, "No video selected to process", Toast.LENGTH_SHORT).show()
            }
        }
        loadPyTorchModel()
    }


    private fun setupDeviceSpinner() {
        val devices = arrayOf("CPU", "GPU (Vulkan)", "NNAPI") // Added NNAPI
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, devices)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerDeviceSelection.adapter = adapter

        spinnerDeviceSelection.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                val newSelectedDeviceEnum = when (position) {
                    1 -> Device.VULKAN
                    else -> Device.CPU
                }

                if (newSelectedDeviceEnum != selectedDevice || pytorchModule == null) {
                    selectedDevice = newSelectedDeviceEnum
                    Log.i("MainActivity", "Device selection changed to: $selectedDevice. Re-loading model.")
                    pytorchModule?.destroy() // Release previous module
                    pytorchModule = null
                    currentModuleDevice = null
                    loadPyTorchModel() // Load with new device
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>) {}
        }
        // Set default selection (CPU), this will also trigger onItemSelected and initial model load
        spinnerDeviceSelection.setSelection(0)
    }

    // Modify or add a new save function:
    private fun saveBitmapToSpecificDirectory(context: Context, bitmap: Bitmap, directory: File, filename: String): String? {
        val file = File(directory, filename)
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.i("FileSave", "Bitmap saved to ${file.absolutePath}")
            return file.absolutePath
        } catch (e: IOException) {
            Log.e("FileSave", "Error saving bitmap to ${file.absolutePath}", e)
        }
        return null
    }

    private fun resizeAndCenterPad(source: Bitmap, targetWidth: Int, targetHeight: Int, backgroundColor: Int = android.graphics.Color.BLACK): Bitmap {
        val resultBitmap = Bitmap.createBitmap(targetWidth, targetHeight, source.config ?: Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultBitmap)
        canvas.drawColor(backgroundColor) // Fill with background color

        val sourceWidth = source.width
        val sourceHeight = source.height

        // Calculate scaling factor to fit the source bitmap into target dimensions
        val xScale = targetWidth.toFloat() / sourceWidth
        val yScale = targetHeight.toFloat() / sourceHeight
        val scale = xScale.coerceAtMost(yScale) // Use the smaller scale factor (fit behavior)

        val scaledWidth = scale * sourceWidth
        val scaledHeight = scale * sourceHeight

        // Calculate position to center the scaled image
        val left = (targetWidth - scaledWidth) / 2f
        val top = (targetHeight - scaledHeight) / 2f

        val destRect = android.graphics.RectF(left, top, left + scaledWidth, top + scaledHeight)
        canvas.drawBitmap(source, null, destRect, null) // Draw source onto the result bitmap, scaled and centered

        return resultBitmap
    }

    // --- Video Processing Core Logic ---
    private suspend fun processVideo(videoUri: Uri) {
        Log.d("ProcessVideo", "Starting video processing for URI: $videoUri on $selectedDevice with Coroutine Channels")

        // Channels for communication between coroutines
        val preprocessedFrameChannel = Channel<Bitmap>(PREPROCESSED_FRAME_CHANNEL_CAPACITY)
        val inferenceInputChannel = Channel<FrameTriplet>(INFERENCE_INPUT_CHANNEL_CAPACITY)
        val processedBitmapChannel = Channel<InferenceOutput>(PROCESSED_BITMAP_CHANNEL_CAPACITY)

        var videoProcessingJob: Job? = null
        var frameConsumptionJob: Job? = null
        var inferenceJob: Job? = null
        var savingJob: Job? = null

        try {
            // Launch a parent job for all processing steps to manage cancellation
            videoProcessingJob = lifecycleScope.launch(Dispatchers.IO) {
                val extractor = MediaExtractor()
                var decoder: MediaCodec? = null
                var totalFramesEstimate = 0
                var frameRate = 30

                try {
                    // --- Stage 1: Frame Extraction and Preprocessing Coroutine ---
                    launch {
                        Log.d("ProcessVideo", "[DecoderCoroutine] Starting")
                        try {
                            extractor.setDataSource(this@MainActivity, videoUri, null)
                            var trackIndex = -1
                            var videoFormat: MediaFormat? = null
                            for (i in 0 until extractor.trackCount) {
                                val format = extractor.getTrackFormat(i)
                                val mime = format.getString(MediaFormat.KEY_MIME)
                                if (mime?.startsWith("video/") == true) {
                                    videoFormat = format
                                    trackIndex = i
                                    extractor.selectTrack(trackIndex)
                                    break
                                }
                            }

                            if (trackIndex == -1 || videoFormat == null) {
                                updateUiOnError("No video track found.")
                                preprocessedFrameChannel.close(IllegalStateException("No video track"))
                                return@launch
                            }

                            val originalWidth = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
                            val originalHeight = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)
                            val durationUs = videoFormat.getLong(MediaFormat.KEY_DURATION)
                            frameRate = videoFormat.getInteger(MediaFormat.KEY_FRAME_RATE, 30)
                            totalFramesEstimate = (durationUs / 1000000 * frameRate).toInt()

                            updateUiStatus("Video: ${originalWidth}x${originalHeight}, ${durationUs / 1000000}s, $frameRate FPS")

                            val mimeType = videoFormat.getString(MediaFormat.KEY_MIME)!!
                            decoder = MediaCodec.createDecoderByType(mimeType)
                            decoder!!.configure(videoFormat, null, null, 0)
                            decoder!!.start()

                            val bufferInfo = MediaCodec.BufferInfo()
                            var isDecoderEOS = false
                            var decodedFrameCount = 0

                            while (!isDecoderEOS && isActive) {
                                val inputBufferIndex = decoder!!.dequeueInputBuffer(10000) // 10ms
                                if (inputBufferIndex >= 0) {
                                    val inputBuffer = decoder!!.getInputBuffer(inputBufferIndex)
                                    val sampleSize = extractor.readSampleData(inputBuffer!!, 0)
                                    if (sampleSize < 0) {
                                        decoder!!.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                        isDecoderEOS = true
                                    } else {
                                        decoder!!.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.sampleTime, 0)
                                        extractor.advance()
                                    }
                                }

                                val outputBufferIndex = decoder!!.dequeueOutputBuffer(bufferInfo, 10000) // 10ms
                                if (outputBufferIndex >= 0) {
                                    val image = decoder!!.getOutputImage(outputBufferIndex)
                                    if (image != null) {
                                        val originalFrameBitmap = imageToBitmap(image) // Handles image.close()
                                        val modelInputLrBitmap = resizeAndCenterPad(originalFrameBitmap, TARGET_LR_WIDTH_FOR_MODEL, TARGET_LR_HEIGHT)
                                        originalFrameBitmap.recycle()
                                        preprocessedFrameChannel.send(modelInputLrBitmap) // Send to next stage
                                        decodedFrameCount++
                                    }
                                    decoder!!.releaseOutputBuffer(outputBufferIndex, false)
                                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                                        isDecoderEOS = true
                                    }
                                }
                            }
                            Log.d("ProcessVideo", "[DecoderCoroutine] Finished. Decoded $decodedFrameCount frames.")
                        } catch (e: Exception) {
                            Log.e("ProcessVideo", "[DecoderCoroutine] Error", e)
                            preprocessedFrameChannel.close(e)
                        } finally {
                            decoder?.stop()
                            decoder?.release()
                            extractor.release()
                            preprocessedFrameChannel.close() // Signal end of frames
                            Log.d("ProcessVideo", "[DecoderCoroutine] Cleaned up and closed channel.")
                        }
                    } // End of Decoder Coroutine

                    // --- Stage 2: Frame Triplet Formation Coroutine ---
                    frameConsumptionJob = launch {
                        Log.d("ProcessVideo", "[TripletFormationCoroutine] Starting")
                        val lrFrameBuffer = mutableListOf<Bitmap>()
                        var frameCounter = 0
                        try {
                            for (bitmap in preprocessedFrameChannel) { // Consume from channel
                                if (!isActive) break
                                lrFrameBuffer.add(bitmap)
                                if (lrFrameBuffer.size == 3) {
                                    val triplet = FrameTriplet(lrFrameBuffer[0], lrFrameBuffer[1], lrFrameBuffer[2], frameCounter - 2) // originalFrameIndex is for the middle frame
                                    inferenceInputChannel.send(triplet)
                                    lrFrameBuffer.removeAt(0) // Keep buffer sliding (don't recycle here, inference needs it)
                                }
                                frameCounter++
                            }
                            // Handle remaining frames at EOS if any (padding)
                            if (lrFrameBuffer.isNotEmpty() && lrFrameBuffer.size < 3) {
                                Log.d("ProcessVideo", "[TripletFormationCoroutine] Padding last triplet.")
                                val lastFrame = lrFrameBuffer.last()
                                while (lrFrameBuffer.size < 3 && lrFrameBuffer.isNotEmpty()) {
                                    lrFrameBuffer.add(lastFrame.copy(lastFrame.config ?: Bitmap.Config.ARGB_8888, true))
                                }
                                if(lrFrameBuffer.size == 3){
                                    val triplet = FrameTriplet(lrFrameBuffer[0], lrFrameBuffer[1], lrFrameBuffer[2], frameCounter - lrFrameBuffer.size)
                                    inferenceInputChannel.send(triplet)
                                }
                            }
                            Log.d("ProcessVideo", "[TripletFormationCoroutine] Finished. Processed $frameCounter input frames.")
                        } catch (e: Exception) {
                            Log.e("ProcessVideo", "[TripletFormationCoroutine] Error", e)
                            inferenceInputChannel.close(e)
                        } finally {
                            inferenceInputChannel.close()
                            // Recycle any remaining bitmaps in lrFrameBuffer if not sent
                            lrFrameBuffer.forEach { if (!it.isRecycled) it.recycle() }
                            Log.d("ProcessVideo", "[TripletFormationCoroutine] Cleaned up and closed channel.")
                        }
                    } // End of Frame Triplet Formation Coroutine

                    // Helper function within processVideo scope to handle batch processing logic
                    suspend fun CoroutineScope.processBatch(
                        batch: List<FrameTriplet>,
                        totalFramesEstimate: Int,
                        tripletsProcessed: Int // Renamed to avoid conflict
                    ) {
                        if (batch.isEmpty() || !isActive) return

                        updateUiStatus("Inferring batch of ${batch.size} (total processed: $tripletsProcessed)...")

                        val mean = floatArrayOf(0.5f, 0.5f, 0.5f)
                        val std = floatArrayOf(0.5f, 0.5f, 0.5f)
                        val h = TARGET_LR_HEIGHT
                        val w = TARGET_LR_WIDTH_FOR_MODEL
                        val channelsPerTriplet = 9
                        val numImagesInBatch = batch.size

                        // Combined float array for the entire batch
                        val batchCombinedFloats = FloatArray(numImagesInBatch * channelsPerTriplet * h * w)
                        var currentBatchDestPos = 0

                        batch.forEachIndexed { index, triplet ->
                            val tensor1 = TensorImageUtils.bitmapToFloat32Tensor(triplet.f1, mean, std)
                            val tensor2 = TensorImageUtils.bitmapToFloat32Tensor(triplet.f2, mean, std)
                            val tensor3 = TensorImageUtils.bitmapToFloat32Tensor(triplet.f3, mean, std)

                            // Recycle f1 of each triplet. f2 and f3 are managed by TripletFormationCoroutine's finally block.
                            if (!triplet.f1.isRecycled) triplet.f1.recycle()
                            // DO NOT recycle triplet.f2 or triplet.f3 here as they might be f1/f2 of the *next* triplet in the list
                            // if INFERENCE_BATCH_SIZE > 1 and they were part of the lrFrameBuffer sliding window.
                            // The TripletFormationCoroutine's finally block is the safest place for f2, f3 from the original buffer.

                            val tripletFloats1 = tensor1.dataAsFloatArray
                            val tripletFloats2 = tensor2.dataAsFloatArray
                            val tripletFloats3 = tensor3.dataAsFloatArray

                            // Concatenate channels for one triplet
                            System.arraycopy(tripletFloats1, 0, batchCombinedFloats, currentBatchDestPos, tripletFloats1.size)
                            currentBatchDestPos += tripletFloats1.size
                            System.arraycopy(tripletFloats2, 0, batchCombinedFloats, currentBatchDestPos, tripletFloats2.size)
                            currentBatchDestPos += tripletFloats2.size
                            System.arraycopy(tripletFloats3, 0, batchCombinedFloats, currentBatchDestPos, tripletFloats3.size)
                            currentBatchDestPos += tripletFloats3.size
                        }

                        val inputTensor = Tensor.fromBlob(batchCombinedFloats, longArrayOf(numImagesInBatch.toLong(), channelsPerTriplet.toLong(), h.toLong(), w.toLong()))

                        val startTime = System.nanoTime()
                        val outputBatchTensor = pytorchModule!!.forward(IValue.from(inputTensor)).toTensor()
                        val inferenceDurationMs = (System.nanoTime() - startTime) / 1_000_000.0
                        Log.d("Performance", "Inference for batch of $numImagesInBatch on $selectedDevice: $inferenceDurationMs ms (avg: ${inferenceDurationMs/numImagesInBatch} ms/triplet)")

                        val outputBatchFloatTensor = outputBatchTensor.dataAsFloatArray
                        // Expected shape: [numImagesInBatch, 3, H_out, W_out]
                        val outChannels = 3
                        val outHeight = TARGET_LR_HEIGHT * UPSCALE_FACTOR
                        val outWidth = TARGET_LR_WIDTH_FOR_MODEL * UPSCALE_FACTOR
                        val floatsPerHrImage = outChannels * outHeight * outWidth

                        for (i in 0 until numImagesInBatch) {
                            if (!isActive) break // Check for cancellation before processing each item in batch

                            val triplet = batch[i]
                            val outputHrBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)
                            val intPixels = IntArray(outWidth * outHeight)
                            val hrImageFloatOffset = i * floatsPerHrImage

                            for (y in 0 until outHeight) {
                                for (x in 0 until outWidth) {
                                    val rIdx = hrImageFloatOffset + (0 * outHeight + y) * outWidth + x
                                    val gIdx = hrImageFloatOffset + (1 * outHeight + y) * outWidth + x
                                    val bIdx = hrImageFloatOffset + (2 * outHeight + y) * outWidth + x

                                    val r = ((outputBatchFloatTensor[rIdx] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)
                                    val g = ((outputBatchFloatTensor[gIdx] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)
                                    val b = ((outputBatchFloatTensor[bIdx] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)
                                    intPixels[y * outWidth + x] = android.graphics.Color.rgb(r, g, b)
                                }
                            }
                            outputHrBitmap.setPixels(intPixels, 0, outWidth, 0, 0, outWidth, outHeight)
                            processedBitmapChannel.send(InferenceOutput(outputHrBitmap, triplet.originalFrameIndex))

                            if (totalFramesEstimate > 0) {
                                val currentTotalProcessed = tripletsProcessed + i + 1
                                updateUiProgress((currentTotalProcessed * 100) / (totalFramesEstimate - 2).coerceAtLeast(1))
                            }
                        }
                    }

                    // --- Stage 3: Inference Coroutine ---
                    inferenceJob = launch(Dispatchers.Default) { // Can use Dispatchers.Default or a dedicated context for compute
                        Log.d("ProcessVideo", "[InferenceCoroutine] Starting on $selectedDevice with batch size $INFERENCE_BATCH_SIZE")
                        var totalTripletsProcessed = 0
                        val tripletBatch = mutableListOf<FrameTriplet>()

                        try {
                            for (triplet in inferenceInputChannel) { // Consume from channel
                                if (!isActive) break
                                tripletBatch.add(triplet)

                                if (tripletBatch.size >= INFERENCE_BATCH_SIZE) {
                                    processBatch(tripletBatch, totalFramesEstimate, tripletsProcessed = totalTripletsProcessed)
                                    totalTripletsProcessed += tripletBatch.size
                                    tripletBatch.clear()
                                }
                            }
                            // Process any remaining triplets in the batch after the channel is closed
                            if (tripletBatch.isNotEmpty() && isActive) {
                                Log.d("ProcessVideo", "[InferenceCoroutine] Processing remaining ${tripletBatch.size} triplets.")
                                processBatch(tripletBatch, totalFramesEstimate, tripletsProcessed = totalTripletsProcessed)
                                totalTripletsProcessed += tripletBatch.size
                                tripletBatch.clear()
                            }
                            Log.d("ProcessVideo", "[InferenceCoroutine] Finished. Processed $totalTripletsProcessed triplets in total.")
                        } catch (e: Exception) {
                            Log.e("ProcessVideo", "[InferenceCoroutine] Error", e)
                            processedBitmapChannel.close(e)
                            // Recycle any bitmaps in a pending batch if an error occurs
                            tripletBatch.forEach { t ->
                                if (!t.f1.isRecycled) t.f1.recycle()
                                if (!t.f2.isRecycled) t.f2.recycle()
                                if (!t.f3.isRecycled) t.f3.recycle()
                            }
                        } finally {
                            processedBitmapChannel.close()
                            Log.d("ProcessVideo", "[InferenceCoroutine] Cleaned up and closed channel.")
                        }
                    } // End of Inference Coroutine

                    // --- Stage 4: Saving/Display Coroutine ---
                    savingJob = launch {
                        Log.d("ProcessVideo", "[SavingCoroutine] Starting")
                        val outputDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "LMF_CNN_Output_Parallel")
                        if (!outputDir.exists()) outputDir.mkdirs()
                        var savedFrameCount = 0
                        val receivedFrames = mutableListOf<InferenceOutput>()

                        try {
                            for (output in processedBitmapChannel) { // Consume from channel
                                if (!isActive) break
                                receivedFrames.add(output)
                            }
                            // Sort frames by original index before saving to maintain order
                            receivedFrames.sortBy { it.originalFrameIndex }

                            updateUiStatus("Saving ${receivedFrames.size} processed HR frames...")
                            receivedFrames.forEach { inferenceOut ->
                                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                                val fileName = "hr_frame_${timestamp}_${String.format("%04d", inferenceOut.originalFrameIndex)}.png"
                                saveBitmapToSpecificDirectory(this@MainActivity, inferenceOut.hrBitmap, outputDir, fileName)
                                inferenceOut.hrBitmap.recycle() // Recycle after saving
                                savedFrameCount++
                            }
                            updateUiStatus("$savedFrameCount HR frames saved to ${outputDir.name}.")
                            Log.d("ProcessVideo", "[SavingCoroutine] Finished. Saved $savedFrameCount frames.")
                        } catch (e: Exception) {
                            Log.e("ProcessVideo", "[SavingCoroutine] Error", e)
                        } finally {
                            // Ensure any remaining bitmaps are recycled if an error occurred before saving all
                            receivedFrames.forEach { if (!it.hrBitmap.isRecycled) it.hrBitmap.recycle() } // Fixed: Call recycle on hrBitmap
                            Log.d("ProcessVideo", "[SavingCoroutine] Cleaned up.")
                        }
                    } // End of Saving Coroutine

                    // Wait for all stages to complete
                    // The individual coroutines launched within this scope will be joined automatically
                    // when the parent `videoProcessingJob` (this launch block) completes.

                } catch (e: Exception) {
                    Log.e("ProcessVideo", "Error in main processing setup", e)
                    updateUiOnError("Error: ${e.localizedMessage}")
                    // Ensure channels are closed if an error happens before coroutines are launched properly
                    preprocessedFrameChannel.close(e)
                    inferenceInputChannel.close(e)
                    processedBitmapChannel.close(e)
                } finally {
                    Log.d("ProcessVideo", "All processing stages launched. Waiting for completion.")
                }
            } // End of videoProcessingJob (parent launch)

            videoProcessingJob?.join() // Wait for the entire pipeline to complete
            Log.i("ProcessVideo", "Video processing fully completed.")
            updateUiStatus("Video processing finished!")

        } catch (e: Exception) {
            Log.e("ProcessVideo", "Outer error in processVideo coroutine with channels", e)
            updateUiOnError("General Error: ${e.localizedMessage}")
        } finally {
            // Ensure UI is re-enabled
            withContext(Dispatchers.Main) {
                progressBar.visibility = View.GONE
                buttonProcessVideo.isEnabled = true
                buttonSelectVideo.isEnabled = true
            }
        }
    }

    // --- UI Update Helper Functions (run on Main thread) ---
    private suspend fun updateUiStatus(message: String) {
        withContext(Dispatchers.Main) {
            textViewStatus.text = message
            Log.i("UIUpdate", message)
        }
    }

    private suspend fun updateUiProgress(progress: Int) {
        withContext(Dispatchers.Main) {
            progressBar.progress = progress
        }
    }

    private suspend fun updateUiOnError(errorMessage: String) {
        withContext(Dispatchers.Main) {
            textViewStatus.text = errorMessage
            progressBar.visibility = View.GONE
            buttonProcessVideo.isEnabled = true
            buttonSelectVideo.isEnabled = true
            Toast.makeText(this@MainActivity, errorMessage, Toast.LENGTH_LONG).show()
        }
    }

    // --- Bitmap Utility Functions ---
    private fun resizeBitmap(source: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createScaledBitmap(source, newWidth, newHeight, true)
    }

    // Placeholder for YUV Image to Bitmap conversion
    // IMPORTANT: This is a simplified placeholder. A real implementation is needed for YUV to RGB.
    private fun imageToBitmap(image: Image): Bitmap {
        val width = image.width
        val height = image.height

        if (image.format == android.graphics.ImageFormat.YUV_420_888) {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 90, out)
            val imageBytes = out.toByteArray()
            image.close()
            return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } else {
            Log.e("imageToBitmap", "Unsupported image format: ${image.format}. Needs specific handling.")
            if (image.format == android.graphics.ImageFormat.FLEX_RGBA_8888 || image.planes.size == 1) {
                val buffer = image.planes[0].buffer
                val pixelStride = image.planes[0].pixelStride
                val rowStride = image.planes[0].rowStride
                val rowPadding = rowStride - pixelStride * width
                val bitmap = Bitmap.createBitmap(width + rowPadding / pixelStride, height, Bitmap.Config.ARGB_8888)
                bitmap.copyPixelsFromBuffer(buffer)
                image.close()
                if (rowPadding > 0) {
                    val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, width, height)
                    bitmap.recycle()
                    return croppedBitmap
                }
                return bitmap
            }
            image.close()
            val errorBitmap = Bitmap.createBitmap(TARGET_LR_WIDTH_FOR_MODEL, TARGET_LR_HEIGHT, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(errorBitmap)
            canvas.drawColor(android.graphics.Color.RED)
            return errorBitmap
        }
    }

    private fun saveBitmapToDownloads(context: Context, bitmap: Bitmap, filename: String): String? {
        val directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        if (!directory.exists()) {
            directory.mkdirs()
        }
        val file = File(directory, filename)
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            Log.i("FileSave", "Bitmap saved to ${file.absolutePath}")
            return file.absolutePath
        } catch (e: IOException) {
            Log.e("FileSave", "Error saving bitmap", e)
        }
        return null
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    private fun loadPyTorchModel() {
        if (pytorchModule != null && currentModuleDevice == selectedDevice) {
            Log.i("MainActivity", "Model already loaded for $selectedDevice.")
            textViewStatus.text = "Model ready on $selectedDevice."
            buttonProcessVideo.isEnabled = selectedVideoUri != null
            return
        }

        pytorchModule?.destroy()
        pytorchModule = null

        Log.i("MainActivity", "Attempting to load PyTorch model on $selectedDevice...")
        textViewStatus.text = "Loading model on $selectedDevice..."
        buttonProcessVideo.isEnabled = false
        buttonSelectVideo.isEnabled = false

        lifecycleScope.launch(Dispatchers.IO) {
            var success = false
            var errorMessage: String? = null
            try {
                val modelPath = assetFilePath(this@MainActivity, modelAssetName)
                pytorchModule = Module.load(modelPath, mutableMapOf<String, String>(), selectedDevice)
                currentModuleDevice = selectedDevice
                success = true
            } catch (e: Exception) {
                currentModuleDevice = null
                errorMessage = e.message
                Log.e("MainActivity", "Error loading PyTorch model on $selectedDevice", e)
            }

            withContext(Dispatchers.Main) {
                buttonSelectVideo.isEnabled = true
                if (success) {
                    Log.i("MainActivity", "PyTorch model loaded successfully on $selectedDevice.")
                    textViewStatus.text = "Model loaded on $selectedDevice. Ready."
                    buttonProcessVideo.isEnabled = selectedVideoUri != null
                } else {
                    textViewStatus.text = "Error loading on $selectedDevice: ${errorMessage ?: "Unknown error"}"
                    Toast.makeText(this@MainActivity, "Model loading failed on $selectedDevice: ${errorMessage ?: "Unknown"}", Toast.LENGTH_LONG).show()
                    buttonProcessVideo.isEnabled = false

                    if (selectedDevice != Device.CPU) {
                        Log.w("MainActivity", "Falling back to CPU due to error on $selectedDevice.")
                        Toast.makeText(this@MainActivity, "Falling back to CPU.", Toast.LENGTH_SHORT).show()

                        selectedDevice = Device.CPU
                        spinnerDeviceSelection.setSelection(0, false)
                        loadPyTorchModel()
                    } else {
                        textViewStatus.text = "Failed to load model on CPU as well."
                    }
                }
            }
        }
    }

    private fun checkPermissionAndOpenVideoPicker() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_VIDEO
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            Manifest.permission.READ_EXTERNAL_STORAGE
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        val writePermission = Manifest.permission.WRITE_EXTERNAL_STORAGE

        val permissionsToRequest = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(permission)
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(this, writePermission) != PackageManager.PERMISSION_GRANTED) {
            permissionsToRequest.add(writePermission)
        }

        if (permissionsToRequest.isEmpty()) {
            openVideoPicker()
        } else {
            multiplePermissionsLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    private val multiplePermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            var allGranted = true
            permissions.entries.forEach {
                if (!it.value) {
                    allGranted = false
                    Log.e("Permissions", "Permission denied: ${it.key}")
                }
            }
            if (allGranted) {
                Log.i("Permissions", "All required permissions granted.")
                openVideoPicker()
            } else {
                Toast.makeText(this, "Storage and/or Media permissions are required.", Toast.LENGTH_LONG).show()
            }
        }

    private val pickVideoLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    selectedVideoUri = uri
                    textViewSelectedVideo.text = "Selected: ${getFileName(uri)}"
                    buttonProcessVideo.isEnabled = pytorchModule != null
                    Log.i("MainActivity", "Video selected: $uri")
                } ?: run {
                    textViewSelectedVideo.text = "Failed to get video URI"
                    buttonProcessVideo.isEnabled = false
                }
            } else {
                Log.i("MainActivity", "Video selection cancelled or failed")
            }
        }

    private fun openVideoPicker() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
        pickVideoLauncher.launch(intent)
    }

    private fun getFileName(uri: Uri): String {
        var fileName: String? = null
        if (uri.scheme.equals("content")) {
            val cursor = contentResolver.query(uri, null, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    val displayNameIndex = it.getColumnIndex(MediaStore.Video.Media.DISPLAY_NAME)
                    if (displayNameIndex != -1) {
                        fileName = it.getString(displayNameIndex)
                    }
                }
            }
        }
        if (fileName == null) {
            fileName = uri.path
            val cut = fileName?.lastIndexOf('/')
            if (cut != -1 && cut != null) {
                fileName = fileName?.substring(cut + 1)
            }
        }
        return fileName ?: "Unknown Video"
    }

    override fun onDestroy() {
        super.onDestroy()
        pytorchModule?.destroy()
        pytorchModule = null
    }
}
