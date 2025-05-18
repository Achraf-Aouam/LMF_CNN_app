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
//import org.pytorch.LiteModuleLoader

// Constants for video processing
private const val TARGET_LR_HEIGHT = 120
// Width will be calculated based on aspect ratio, but let's assume a common one for now
// Or better, calculate it from the source video.
// For the model, it expects 214 width for 120 height.
private const val TARGET_LR_WIDTH_FOR_MODEL = 214
private const val UPSCALE_FACTOR = 4 // From your model

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
        val devices = arrayOf("CPU", "GPU (Vulkan)" )
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, devices)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerDeviceSelection.adapter = adapter

        spinnerDeviceSelection.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                val newSelectedDeviceEnum = when (position) {
                    1 -> Device.VULKAN
//                    2 -> Device.NNAPI
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

    // --- Video Processing Core Logic (will be expanded) ---
    private suspend fun processVideo(videoUri: Uri) {
        Log.d("ProcessVideo", "Starting video processing for URI: $videoUri  on $selectedDevice")
        var lrVideoPath: String? = null
        var hrVideoPath: String? = null

        try {
            // This will run on a background thread (Dispatchers.IO)
            withContext(Dispatchers.IO) {
                val retriever = MediaMetadataRetriever()
                try {
                    retriever.setDataSource(this@MainActivity, videoUri)

                    val originalWidthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                    val originalHeightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                    val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    val frameRateStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE) // May be null

                    if (originalWidthStr == null || originalHeightStr == null || durationStr == null) {
                        Log.e("ProcessVideo", "Failed to extract video metadata.")
                        updateUiOnError("Failed to get video metadata.")
                        return@withContext
                    }

                    val originalWidth = originalWidthStr.toInt()
                    val originalHeight = originalHeightStr.toInt()
                    val durationMs = durationStr.toLong()
                    // Estimate frame rate if not available, default to 25 or 30
                    val frameRate = frameRateStr?.toFloatOrNull() ?: 30f
                    val frameIntervalUs = (1_000_000 / frameRate).toLong() // Microseconds

                    Log.i("ProcessVideo", "Original Video: ${originalWidth}x${originalHeight}, Duration: ${durationMs}ms, FPS: $frameRate")
                    updateUiStatus("Video Info: ${originalWidth}x${originalHeight}, ${durationMs / 1000}s")

                    // Calculate LR dimensions maintaining aspect ratio, targeting TARGET_LR_HEIGHT
                    val aspectRatio = originalWidth.toFloat() / originalHeight.toFloat()
                    val lrHeight = TARGET_LR_HEIGHT
                    val lrWidth = (lrHeight * aspectRatio).roundToInt()
                    Log.i("ProcessVideo", "Target LR dimensions (aspect preserved): ${lrWidth}x${lrHeight}")


                    // --- Frame Buffer for Model Input ---
                    // Model expects 3 consecutive frames. We'll use a circular buffer or list.
                    val lrFrameBuffer = mutableListOf<Bitmap>()
                    val processedHrFrames = mutableListOf<Bitmap>() // To store output HR frames

                    var currentFrameTimeUs = 0L
                    var framesProcessed = 0
                    val totalFramesEstimate = (durationMs / 1000 * frameRate).toInt()

                    // Loop to extract frames
                    // MediaMetadataRetriever.getFrameAtTime is okay for a few frames, but inefficient for many.
                    // For full video, MediaCodec/MediaExtractor is better.
                    // We'll simulate frame-by-frame processing for now.

                    // --- Part 1: Frame Extraction, Downscaling, and Input Tensor Preparation ---
                    while (currentFrameTimeUs < durationMs * 1000) {
                        // Extract frame
                        // OPTION_CLOSEST_SYNC is faster but less accurate. OPTION_CLOSEST for more precision.
                        val originalFrameBitmap = retriever.getFrameAtTime(currentFrameTimeUs, MediaMetadataRetriever.OPTION_CLOSEST)
                        if (originalFrameBitmap == null) {
                            Log.w("ProcessVideo", "Could not retrieve frame at ${currentFrameTimeUs / 1000}ms. Skipping.")
                            currentFrameTimeUs += frameIntervalUs
                            if (currentFrameTimeUs > durationMs * 1000 && lrFrameBuffer.size < 3 && lrFrameBuffer.isNotEmpty()) {
                                // If near end and buffer not full, pad with last frame (simplistic)
                                Log.d("ProcessVideo", "Padding buffer with last frame at end of video")
                                while(lrFrameBuffer.size < 3 && lrFrameBuffer.isNotEmpty()) {
                                    lrFrameBuffer.add(lrFrameBuffer.last().copy(lrFrameBuffer.last().config ?: Bitmap.Config.ARGB_8888 , true))
                                }
                            } else if (originalFrameBitmap == null && currentFrameTimeUs > durationMs * 1000) {
                                break // End of video
                            }
                            continue
                        }

                        // Downscale frame to LR (e.g., 214x120 for the model)
                        // Important: The model was trained on LR images of a specific size.
                        // We need to resize AND potentially crop to match 214x120.
                        // Let's resize to lrHeight (120) maintaining aspect, then center crop to TARGET_LR_WIDTH_FOR_MODEL (214).
                        val resizedLrBitmap = resizeBitmap(originalFrameBitmap, lrWidth, lrHeight)
                        val modelInputLrBitmap = resizeAndCenterPad(originalFrameBitmap!!, TARGET_LR_WIDTH_FOR_MODEL, TARGET_LR_HEIGHT)
                        originalFrameBitmap.recycle() // Important to free memory
                        resizedLrBitmap.recycle() // If different from modelInputLrBitmap

                        lrFrameBuffer.add(modelInputLrBitmap)

                        // If we have 3 frames in the buffer, process them
                        if (lrFrameBuffer.size == 3) {
                            updateUiStatus("Processing frame triplet ${framesProcessed + 1}...")
                            // --- Part 2: Preprocess for Model & Inference ---
                            // Concatenate 3 LR frames (Bitmaps) into one 9-channel Tensor
                            // Normalization: [-1, 1] as per your training
                            val mean = floatArrayOf(0.5f, 0.5f, 0.5f)
                            val std = floatArrayOf(0.5f, 0.5f, 0.5f)

                            // Convert each bitmap to a 3-channel tensor and then stack
                            val tensor1 = TensorImageUtils.bitmapToFloat32Tensor(lrFrameBuffer[0], mean, std)
                            val tensor2 = TensorImageUtils.bitmapToFloat32Tensor(lrFrameBuffer[1], mean, std)
                            val tensor3 = TensorImageUtils.bitmapToFloat32Tensor(lrFrameBuffer[2], mean, std)

                            // Input tensor shape should be [1, 9, H, W]
                            // H = TARGET_LR_HEIGHT (120), W = TARGET_LR_WIDTH_FOR_MODEL (214)
//                            val inputTensor = Tensor.cat(arrayOf(tensor1, tensor2, tensor3), 1) // Concatenate along channel dim
//                                .reshape(1, 9, TARGET_LR_HEIGHT, TARGET_LR_WIDTH_FOR_MODEL)

                            val h = TARGET_LR_HEIGHT
                            val w = TARGET_LR_WIDTH_FOR_MODEL
                            val cPerTensor = 3 // Channels per individual frame tensor
                            val numTensors = 3 // Number of frames to stack

                            val totalChannels = cPerTensor * numTensors // Should be 9

// Get data from individual tensors
                            val floats1 = tensor1.dataAsFloatArray // Expected size: 1 * 3 * H * W
                            val floats2 = tensor2.dataAsFloatArray
                            val floats3 = tensor3.dataAsFloatArray

// Create a new float array to hold the combined data
                            val combinedFloats = FloatArray(1 * totalChannels * h * w) // Batch_size = 1

                            var destPos = 0
// Copy data from tensor1 (all its channels)
                            System.arraycopy(floats1, 0, combinedFloats, destPos, floats1.size)
                            destPos += floats1.size
// Copy data from tensor2
                            System.arraycopy(floats2, 0, combinedFloats, destPos, floats2.size)
                            destPos += floats2.size
// Copy data from tensor3
                            System.arraycopy(floats3, 0, combinedFloats, destPos, floats3.size)

                            val inputTensor = Tensor.fromBlob(combinedFloats, longArrayOf(1, totalChannels.toLong(), h.toLong(), w.toLong()))
                            // --- Run Inference ---
                            val startTime = System.nanoTime()
                            val outputTensor = pytorchModule!!.forward(IValue.from(inputTensor)).toTensor()
                            val endTime = System.nanoTime()
                            val durationMs = (endTime - startTime) / 1_000_000.0
                            Log.d("Performance", "Inference for one triplet on $selectedDevice: $durationMs ms")
                            // Output tensor shape: [1, 3, H_out, W_out]
                            // H_out = 120 * 4 = 480, W_out = 214 * 4 = 856

                            // --- Part 3: Postprocess Output Tensor to Bitmap ---
                            // Denormalize from [-1, 1] back to [0, 1] then to Bitmap [0, 255]
                            // For denormalization, if input was (x - 0.5) / 0.5, output is y * 0.5 + 0.5
                            val outputFloatTensor = outputTensor.dataAsFloatArray
                            // Expected output shape: [1, 3, 480, 856]
                            val outChannels = 3
                            val outHeight = TARGET_LR_HEIGHT * UPSCALE_FACTOR
                            val outWidth = TARGET_LR_WIDTH_FOR_MODEL * UPSCALE_FACTOR

                            // Create a new bitmap for the output
                            val outputHrBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888)
                            val intPixels = IntArray(outWidth * outHeight) // Array to hold ARGB pixel values
                            // Manually denormalize and fill bitmap pixels
                            // This is a bit verbose; TensorImageUtils might have a direct way for this too,
                            // but let's be explicit for denormalization.
                            // PyTorch tensor is CHW, Bitmap is row-major (Height then Width)
                            for (y in 0 until outHeight) {
                                for (x in 0 until outWidth) {
                                    // Indices for R, G, B channels in the flat float array (CHW format)
                                    val rIndex = (0 * outHeight + y) * outWidth + x
                                    val gIndex = (1 * outHeight + y) * outWidth + x
                                    val bIndex = (2 * outHeight + y) * outWidth + x

                                    // Denormalize: from [-1, 1] to [0, 1], then to [0, 255]
                                    val r = ((outputFloatTensor[rIndex] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)
                                    val g = ((outputFloatTensor[gIndex] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)
                                    val b = ((outputFloatTensor[bIndex] * 0.5f + 0.5f) * 255f).toInt().coerceIn(0, 255)

                                    intPixels[y * outWidth + x] = android.graphics.Color.rgb(r, g, b) // Or Color.argb(255, r, g, b)
                                }
                            }
                            outputHrBitmap.setPixels(intPixels, 0, outWidth, 0, 0, outWidth, outHeight)
                            processedHrFrames.add(outputHrBitmap)

                            // Remove the oldest frame from the buffer to slide the window
//                            lrFrameBuffer.removeAt(0).recycle() // Recycle the removed bitmap
                            val removedBitmap = lrFrameBuffer.removeAt(0)
                            removedBitmap.recycle() // Recycle the removed bitmap
                            framesProcessed++
                            updateUiProgress((framesProcessed * 100) / totalFramesEstimate)
                        }
                        currentFrameTimeUs += frameIntervalUs
                    } // End of while loop for frames

                    retriever.release() // Release the retriever

                    // --- Part 4: Save LR Video (if desired) & Re-stitch HR Video ---
                    // This part is complex. Saving frame-by-frame extracted bitmaps to a video
                    // requires MediaMuxer and MediaCodec (for encoding).
                    // For simplicity, we'll just log that we have the HR frames.
                    // Actual video saving is a significant step.

                    if (processedHrFrames.isNotEmpty()) {
                        updateUiStatus("Saving ${processedHrFrames.size} processed HR frames...")
                        val outputDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "LMF_CNN_Output")
                        if (!outputDir.exists()) {
                            outputDir.mkdirs()
                        }

                        processedHrFrames.forEachIndexed { index, bmp ->
                            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                            val fileName = "hr_frame_${timestamp}_${String.format("%03d", index)}.png"
                            val savedPath = saveBitmapToSpecificDirectory(this@MainActivity, bmp, outputDir, fileName)
                            if (savedPath != null) {
                                Log.i("ProcessVideo", "Saved HR frame to: $savedPath")
                            } else {
                                Log.e("ProcessVideo", "Failed to save HR frame $index")
                            }
                            // bmp.recycle() // Recycle here if you don't need the bitmaps in memory anymore
                        }
                        updateUiStatus("${processedHrFrames.size} HR frames saved to Downloads/LMF_CNN_Output.")

                        // Clean up processed HR frames if you are done with them
                        processedHrFrames.forEach { it.recycle() }
                        processedHrFrames.clear()

                    } else {
                        updateUiStatus("No frames were processed.")
                    }

                } catch (e: Exception) {
                    Log.e("ProcessVideo", "Error during video processing", e)
                    updateUiOnError("Error: ${e.localizedMessage}")
                } finally {
                    retriever.release()
                }
            } // end withContext(Dispatchers.IO)
        } catch (e: Exception) {
            Log.e("ProcessVideo", "Outer error in processVideo coroutine", e)
            updateUiOnError("General Error: ${e.localizedMessage}")
        } finally {
            // Re-enable UI elements on the main thread
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



    // --- File Saving (Example: Save a Bitmap to Downloads) ---
    // Note: For saving to public directories like Downloads on Android 10+ (API 29+),
    // you should use MediaStore API for better scoped storage compliance.
    // This is a simplified example.
    private fun saveBitmapToDownloads(context: Context, bitmap: Bitmap, filename: String): String? {
        val directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        if (!directory.exists()) {
            directory.mkdirs()
        }
        val file = File(directory, filename)
        try {
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // PNG is lossless
            }
            Log.i("FileSave", "Bitmap saved to ${file.absolutePath}")

            // Optionally, notify MediaScanner
            // MediaScannerConnection.scanFile(context, arrayOf(file.toString()), null, null)

            return file.absolutePath
        } catch (e: IOException) {
            Log.e("FileSave", "Error saving bitmap", e)
        }
        return null
    }

    // Helper function to get absolute path from asset
    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024) // 4k buffer
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
        // If a module is already loaded for the *current* selectedDevice, do nothing.
        if (pytorchModule != null && currentModuleDevice == selectedDevice) {
            Log.i("MainActivity", "Model already loaded for $selectedDevice.")
            textViewStatus.text = "Model ready on $selectedDevice."
            buttonProcessVideo.isEnabled = selectedVideoUri != null
            return
        }

        // If there's an old module, destroy it (should be handled by spinner listener, but good failsafe)
        pytorchModule?.destroy()
        pytorchModule = null

        Log.i("MainActivity", "Attempting to load PyTorch model on $selectedDevice...")
        textViewStatus.text = "Loading model on $selectedDevice..."
        buttonProcessVideo.isEnabled = false // Disable while loading
        buttonSelectVideo.isEnabled = false // Disable while loading

        lifecycleScope.launch(Dispatchers.IO) {
            var success = false
            var errorMessage: String? = null
            try {
                val modelPath = assetFilePath(this@MainActivity, modelAssetName)
                // Use LiteModuleLoader
//                pytorchModule = LiteModuleLoader.load(modelPath, null, selectedDevice)
                pytorchModule = Module.load(modelPath,mutableMapOf<String, String>(), selectedDevice)
                currentModuleDevice = selectedDevice // Store the device it was loaded on
                success = true
            } catch (e: Exception) {
                currentModuleDevice = null
                errorMessage = e.message
                Log.e("MainActivity", "Error loading PyTorch model on $selectedDevice", e)
            }

            withContext(Dispatchers.Main) {
                buttonSelectVideo.isEnabled = true // Re-enable select video button regardless of load outcome
                if (success) {
                    Log.i("MainActivity", "PyTorch model loaded successfully on $selectedDevice.")
                    textViewStatus.text = "Model loaded on $selectedDevice. Ready."
                    buttonProcessVideo.isEnabled = selectedVideoUri != null
                } else {
                    textViewStatus.text = "Error loading on $selectedDevice: ${errorMessage ?: "Unknown error"}"
                    Toast.makeText(this@MainActivity, "Model loading failed on $selectedDevice: ${errorMessage ?: "Unknown"}", Toast.LENGTH_LONG).show()
                    buttonProcessVideo.isEnabled = false

                    // Fallback to CPU if a delegate (GPU/NPU) failed, but not if CPU itself failed
                    if (selectedDevice != Device.CPU) {
                        Log.w("MainActivity", "Falling back to CPU due to error on $selectedDevice.")
                        Toast.makeText(this@MainActivity, "Falling back to CPU.", Toast.LENGTH_SHORT).show()

                        // Update selectedDevice and spinner without triggering listener again to prevent loop
                        selectedDevice = Device.CPU
                        spinnerDeviceSelection.setSelection(0, false)
                        loadPyTorchModel() // Retry with CPU
                    } else {
                        // CPU load failed
                        textViewStatus.text = "Failed to load model on CPU as well."
                    }
                }
            }
        }
    }


private fun checkPermissionAndOpenVideoPicker() {
    val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // API 33+
        Manifest.permission.READ_MEDIA_VIDEO
    } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) { // Android 10 needs read for MediaStore access
        Manifest.permission.READ_EXTERNAL_STORAGE
    }
    else { // Older versions
        Manifest.permission.READ_EXTERNAL_STORAGE
    }


    val writePermission = Manifest.permission.WRITE_EXTERNAL_STORAGE

    val permissionsToRequest = mutableListOf<String>()
    if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
        permissionsToRequest.add(permission)
    }
    // For saving to public Downloads on API < 29, you need WRITE_EXTERNAL_STORAGE
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
        ContextCompat.checkSelfPermission(this, writePermission) != PackageManager.PERMISSION_GRANTED) {
        permissionsToRequest.add(writePermission)
    }


    if (permissionsToRequest.isEmpty()) {
        openVideoPicker()
    } else {
        // Simplified: request all needed at once. Better UX would explain why.
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

    // ActivityResultLauncher for picking a video
    private val pickVideoLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    selectedVideoUri = uri
                    textViewSelectedVideo.text = "Selected: ${getFileName(uri)}"
                    buttonProcessVideo.isEnabled = pytorchModule != null // Enable only if model is also loaded
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
        // You could also use Intent.ACTION_GET_CONTENT and setType("video/*")
        // val intent = Intent(Intent.ACTION_GET_CONTENT)
        // intent.type = "video/*"
        pickVideoLauncher.launch(intent)
    }

    // Helper function to get file name from URI (can be complex, simplified here)
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
        pytorchModule?.destroy() // CRITICAL: Release native resources
        pytorchModule = null
    }
}
