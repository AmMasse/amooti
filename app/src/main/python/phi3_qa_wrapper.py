import onnxruntime_genai as og
import json
import time


class ModelRunner:
    def __init__(self, model_path, execution_provider='follow_config', verbose=False, **search_options):
        self.model_path = model_path
        self.execution_provider = execution_provider
        self.verbose = verbose
        self.search_options = search_options

        # Set default max_length if not provided
        if 'max_length' not in self.search_options:
            self.search_options['max_length'] = 2048

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        if self.verbose:
            print("Loading model...")

        config = og.Config(self.model_path)
        if self.execution_provider != "follow_config":
            config.clear_providers()
            if self.execution_provider != "cpu":
                if self.verbose:
                    print(f"Setting model to {self.execution_provider}")
                config.append_provider(self.execution_provider)

        self.model = og.Model(config)
        if self.verbose:
            print("Model loaded")

        self.tokenizer = og.Tokenizer(self.model)
        if self.verbose:
            print("Tokenizer created")

    def stream_generate(self, input_text, timings=False):
        """Generator that yields tokens as they're generated"""
        # Prepare timings data if enabled
        timings_data = {}
        if timings:
            start_time = time.time()
            first_token_time = None

        # Apply chat template and encode tokens
        messages = [{"role": "user", "content": input_text}]
        input_prompt = self.tokenizer.apply_chat_template(
            messages=json.dumps(messages),
            add_generation_prompt=True
        )
        input_tokens = self.tokenizer.encode(input_prompt)

        # Set up generator
        params = og.GeneratorParams(self.model)
        params.set_search_options(**self.search_options)
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        # Create tokenizer stream
        tokenizer_stream = self.tokenizer.create_stream()
        new_tokens = []

        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]

            # Record first token timing
            if timings and first_token_time is None:
                first_token_time = time.time()

            # Decode and yield token
            decoded_token = tokenizer_stream.decode(token)
            yield decoded_token

            if timings:
                new_tokens.append(token)

        # Calculate timings if enabled
        if timings:
            prompt_time = first_token_time - start_time
            generation_time = time.time() - first_token_time
            timings_data = {
                "prompt_tokens": len(input_tokens),
                "new_tokens": len(new_tokens),
                "time_to_first_token": prompt_time,
                "prompt_tps": len(input_tokens) / prompt_time,
                "new_tokens_tps": len(new_tokens) / generation_time
            }
            yield timings_data

    def interactive_inference(self):
        """Interactive console interface with streaming"""
        print("Model ready for inference. Type 'exit' to quit.")
        while True:
            user_input = input("\nInput: ")
            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                print("Error: Input cannot be empty")
                continue

            print("\nOutput: ", end='', flush=True)
            for token in self.stream_generate(user_input):
                if isinstance(token, dict):  # Handle timings if enabled
                    print(f"\n\nTimings: {token}")
                else:
                    print(token, end='', flush=True)
            print()