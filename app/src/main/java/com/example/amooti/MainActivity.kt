package com.example.amooti

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        val py = Python.getInstance()
        val module = py.getModule("phi3_qa_wrapper")

        val promptInput = findViewById<EditText>(R.id.promptInput)
        val generateBtn = findViewById<Button>(R.id.generateBtn)
        val outputText = findViewById<TextView>(R.id.outputText)

        generateBtn.setOnClickListener {
            val prompt = promptInput.text.toString()
            val result = module.callAttr("stream_generate", prompt).toString()
            outputText.text = result
        }
    }
}
