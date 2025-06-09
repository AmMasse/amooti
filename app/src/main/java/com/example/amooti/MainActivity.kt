package com.example.amooti

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val py = Python.getInstance()
        val promptInput = findViewById<EditText>(R.id.promptInput)
        val generateBtn = findViewById<Button>(R.id.generateBtn)
        val outputText = findViewById<TextView>(R.id.outputText)

        generateBtn.setOnClickListener {
            try {
                val module = py.getModule("phi3_qa_wrapper")
                val modelRunner = module.callAttr("ModelRunner")
                val result = modelRunner.callAttr("run_model", promptInput.text.toString())
                outputText.text = result.toString()
            } catch (e: Exception) {
                outputText.text = "Error: ${e.localizedMessage}"
            }
        }
    }
}
