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



class MainActivity : AppCompatActivity() {

    private lateinit var buttonSelectVideo: Button
    private lateinit var textViewSelectedVideo: TextView
    private lateinit var buttonProcessVideo: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var textViewStatus: TextView

    private var selectedVideoUri: Uri? = null

    private var pytorchModule: Module? = null
    private val modelAssetName = "lmf_cnn_mobile.ptl"

    // ActivityResultLauncher for picking a video
    private val pickVideoLauncher: ActivityResultLauncher<Intent> =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    selectedVideoUri = uri
                    textViewSelectedVideo.text = "Selected: ${getFileName(uri)}"
                    buttonProcessVideo.isEnabled = true
                    Log.i("MainActivity", "Video selected: $uri")
                } ?: run {
                    textViewSelectedVideo.text = "Failed to get video URI"
                    buttonProcessVideo.isEnabled = false
                }
            } else {
                Log.i("MainActivity", "Video selection cancelled or failed")
            }
        }

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
        setContentView(R.layout.activity_main) // Links to your XML layout

        buttonSelectVideo = findViewById(R.id.buttonSelectVideo)
        textViewSelectedVideo = findViewById(R.id.textViewSelectedVideo)
        buttonProcessVideo = findViewById(R.id.buttonProcessVideo)
        progressBar = findViewById(R.id.progressBar)
        textViewStatus = findViewById(R.id.textViewStatus)

        buttonSelectVideo.setOnClickListener {
            checkPermissionAndOpenVideoPicker()
        }

        buttonProcessVideo.setOnClickListener {
            selectedVideoUri?.let { uri ->
                if (pytorchModule == null) {
                    Toast.makeText(this, "Model not loaded yet!", Toast.LENGTH_LONG).show()
                    Log.e("MainActivity", "Process button clicked but model is null.")
                    return@setOnClickListener
                }
                Toast.makeText(this, "Processing: ${getFileName(uri)}", Toast.LENGTH_SHORT).show()
                Log.i("MainActivity", "Start processing video: $uri")
                startVideoProcessing(uri)
            } ?: run {
                Toast.makeText(this, "No video selected to process", Toast.LENGTH_SHORT).show()
            }
        }
        loadPyTorchModel()
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
        Log.i("MainActivity", "Attempting to load PyTorch model...")
        textViewStatus.text = "Loading model..." // Update UI
        try {
            val modelPath = assetFilePath(this, modelAssetName)
            pytorchModule = Module.load(modelPath) // Load the .ptl file
            Log.i("MainActivity", "PyTorch model loaded successfully from: $modelPath")
            textViewStatus.text = "Model loaded. Ready for video selection."
        } catch (e: IOException) {
            Log.e("MainActivity", "Error reading model asset or loading model", e)
            Toast.makeText(this, "Error loading model: ${e.message}", Toast.LENGTH_LONG).show()
            textViewStatus.text = "Error loading model!"
        } catch (e: Exception) { // Catch any other PyTorch specific loading errors
            Log.e("MainActivity", "Error loading PyTorch model", e)
            Toast.makeText(this, "PyTorch model loading failed: ${e.message}", Toast.LENGTH_LONG).show()
            textViewStatus.text = "Error loading model!"
        }
    }

    private fun checkPermissionAndOpenVideoPicker() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // API 33+
            Manifest.permission.READ_MEDIA_VIDEO
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED -> {
                openVideoPicker()
            }
            shouldShowRequestPermissionRationale(permission) -> {
                // Show an explanation to the user *asynchronously*
                // For simplicity, we'll just request again here, but in a real app, show a dialog.
                Toast.makeText(this, "Permission needed to access videos.", Toast.LENGTH_LONG).show()
                requestPermissionLauncher.launch(permission)
            }
            else -> {
                requestPermissionLauncher.launch(permission)
            }
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

    // Placeholder for the actual video processing logic
//    private fun startVideoProcessing(videoUri: Uri) {
//        // This is where the heavy lifting will happen in later steps
//        // For now, just log and update UI
//        textViewStatus.text = "Starting processing..."
//        progressBar.visibility = View.VISIBLE
//        progressBar.isIndeterminate = true // Or set max and update progress later
//
//        // TODO: Next phase - Implement actual video processing logic here
//        // This will involve:
//        // 1. Frame extraction from videoUri
//        // 2. Preprocessing each frame (or triplet of frames) into a Tensor
//        // 3. Running inference: pytorchModule.forward(IValue.from(inputTensor)).toTensor()
//        // 4. Postprocessing the output tensor back to an image/frame
//        // 5. Re-stitching frames into an output video
//        // All of this MUST happen on a background thread (e.g., Kotlin Coroutines)
//
//
//        // Simulate some work for now
//        buttonProcessVideo.isEnabled = false
//        buttonSelectVideo.isEnabled = false
//
//        // In a real app, you'd use Kotlin Coroutines or an AsyncTask/ExecutorService here
//        // For now, a simple Toast to indicate it's a placeholder
//        Toast.makeText(this, "Video processing will happen here for $videoUri", Toast.LENGTH_LONG).show()
//
//        // Example of how you might update UI after (simulated) processing
//        // This would be called from the background thread via runOnUiThread or similar
//        // For now, we'll just reset after a delay to simulate completion
//        progressBar.postDelayed({
//            progressBar.visibility = View.GONE
//            textViewStatus.text = "Processing (simulated) complete!"
//            buttonProcessVideo.isEnabled = true
//            buttonSelectVideo.isEnabled = true
//        }, 5000) // Simulate 5 seconds of work
//    }
//}

    private fun startVideoProcessing(videoUri: Uri) {
        // Check if model is loaded
        if (pytorchModule == null) {
            Log.e("MainActivity", "Model is not loaded. Cannot process video.")
            Toast.makeText(this, "Model not available for processing.", Toast.LENGTH_LONG).show()
            return
        }

        textViewStatus.text = "Starting processing..."
        progressBar.visibility = View.VISIBLE
        progressBar.isIndeterminate = true
        buttonProcessVideo.isEnabled = false
        buttonSelectVideo.isEnabled = false

        // TODO: Next phase - Implement actual video processing logic here
        // This will involve:
        // 1. Frame extraction from videoUri
        // 2. Preprocessing each frame (or triplet of frames) into a Tensor
        // 3. Running inference: pytorchModule.forward(IValue.from(inputTensor)).toTensor()
        // 4. Postprocessing the output tensor back to an image/frame
        // 5. Re-stitching frames into an output video
        // All of this MUST happen on a background thread (e.g., Kotlin Coroutines)

        Log.i("MainActivity", "Video processing logic to be implemented for: $videoUri")

        progressBar.postDelayed({
            progressBar.visibility = View.GONE
            textViewStatus.text = "Processing (simulated) complete!"
            buttonProcessVideo.isEnabled = true
            buttonSelectVideo.isEnabled = true
        }, 5000)
    }
}