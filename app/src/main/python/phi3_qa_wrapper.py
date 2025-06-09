import os
import onnxruntime_genai as og
import json
import time

class ModelRunner:
    def __init__(self, verbose=False, **search_options):
        self.verbose = verbose
        self.search_options = search_options

        if 'max_length' not in self.search_options:
            self.search_options['max_length'] = 2048

        self.model_path = "/sdcard/Download/amooti-model/"
        if not os.path.exists(os.path.join(self.model_path, "model.onnx")):
            raise FileNotFoundError("Model not found in /sdcard/Download/amooti-model/. Please download it.")

        self._initialize_model()

    def _initialize_model(self):
        if self.verbose:
            print("Loading model...")

        config = og.Config(self.model_path)
        self.model = og.Model(config)

        if self.verbose:
            print("Model loaded")

        self.tokenizer = og.Tokenizer(self.model)

    def run_model(self, input_text):
        messages = [{"role": "user", "content": input_text}]
        input_prompt = self.tokenizer.apply_chat_template(
            messages=json.dumps(messages),
            add_generation_prompt=True
        )
        input_tokens = self.tokenizer.encode(input_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        tokenizer_stream = self.tokenizer.create_stream()
        output = ""
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            output += tokenizer_stream.decode(token)
        return output
