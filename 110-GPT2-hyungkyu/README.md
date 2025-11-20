# GPT-2 Implementation with Vulkan Compute Shaders

C++ implementation of GPT-2 (Generative Pre-trained Transformer 2) using Vulkan compute shaders for GPU-accelerated inference.

## Features

- ✅ Full GPT-2 architecture implementation (12 layers, 768 hidden size, 12 attention heads)
- ✅ Vulkan compute shader-based neural network operations
- ✅ BPE (Byte-Pair Encoding) tokenizer
- ✅ Pretrained weight loading from OpenAI checkpoints
- ✅ Text generation with greedy decoding and sampling
- ✅ Configurable generation parameters (temperature, top-k sampling)

## Project Structure

```
110-GPT2-hyungkyu/
├── assets/                          # Data files and model weights
│   ├── vocab.json                   # GPT-2 vocabulary (50,257 tokens)
│   ├── merges.txt                   # BPE merge rules (50,000 rules)
│   ├── the-verdict.txt              # Test data file
│   └── weights/
│       └── 124M/                    # GPT-2 124M model
│           ├── checkpoint           # TensorFlow checkpoint metadata
│           ├── hparams.json         # Model hyperparameters
│           ├── model.ckpt.*         # TensorFlow checkpoint files
│           ├── gpt2_weights.bin     # Binary weights (converted)
│           └── gpt2_config.txt      # Model configuration
├── utils/                           # Utility scripts
│   ├── setup_weights.py             # One-step download + convert (recommended)
│   ├── download_gpt2_weights.py     # Download OpenAI checkpoint
│   └── convert_openai_weights.py    # Convert checkpoint to binary
├── debug/                           # Debugging and verification scripts
│   ├── check_weights.py             # Verify weight loading
│   ├── check_bias_values.py         # Analyze bias magnitudes
│   ├── check_token.py               # Token ID decoder
│   ├── compare_with_pytorch.py      # PyTorch comparison
│   └── ...                          # Other verification scripts
├── core/                            # Core neural network framework
│   ├── neuralNet.h                  # Base neural network class
│   ├── tensor.h                     # Tensor operations
│   └── vulkanApp.h                  # Vulkan compute setup
├── tokenizer/                       # BPE tokenizer implementation
│   ├── tokenizer.h                  # BPE tokenizer class
│   └── test.cpp                     # Tokenizer tests
├── dataloader/                      # Data loading utilities
│   ├── dataloader.h                 # Text data loader
│   └── test.cpp                     # Dataloader tests
├── model/                           # GPT-2 model architecture
│   ├── embedding/                   # Embedding layers
│   ├── attention/                   # Multi-head attention
│   ├── transformerBlock/            # Transformer block
│   ├── gpt2Net.h                    # Main GPT-2 network
│   ├── gpt2Weights.h                # Weight loading
│   ├── gpt2Generation.h             # Text generation
│   └── gpt2Test.cpp                 # Test functions
├── main.cpp                         # Entry point (CLI interface)
└── CMakeLists.txt                   # Build configuration

```

## Prerequisites

### C++ Build Tools
- CMake 3.15+
- C++17 compatible compiler (MSVC, GCC, Clang)
- Vulkan SDK

### Python (for weight conversion)
- Python 3.7+
- Required packages:
  ```bash
  pip install numpy
  pip install tensorflow  # For convert_openai_weights.py
  # OR
  pip install torch transformers  # For download_gpt2_weights.py
  ```

## Setup Instructions

### Step 1: Download and Convert Weights

Download GPT-2 checkpoint from OpenAI and convert to binary format in one command:

```bash
cd 110-GPT2-hyungkyu
python utils/setup_weights.py
```

This script will:
1. Download OpenAI GPT-2 checkpoint files from CDN
2. Convert to binary format using `convert_openai_weights.py`
3. Generate `assets/weights/124M/gpt2_weights.bin` and `gpt2_config.txt`

**Required Python packages:**
```bash
pip install numpy tensorflow
```

**For different model sizes:**
```bash
python utils/setup_weights.py --model 355M --output-dir assets/weights/355M
# Options: 124M (default), 355M, 774M, 1558M
```

**Note:** Tokenizer files (`vocab.json`, `merges.txt`) are already included in the repository under `assets/` folder.

### Step 2: Build the Project

```bash
# From vai-samples root directory
cmake --build build --config Debug --target 110-GPT2-hyungkyu
```

Or for Release build:
```bash
cmake --build build --config Release --target 110-GPT2-hyungkyu
```

### Step 3: Run Tests

```bash
# From vai-samples root directory

# Run with default settings
./bin/debug/110-GPT2-hyungkyu.exe

# Run with custom prompt
./bin/debug/110-GPT2-hyungkyu.exe "Once upon a time"

# Run with custom prompt and token count
./bin/debug/110-GPT2-hyungkyu.exe "Once upon a time" 15

# Show help
./bin/debug/110-GPT2-hyungkyu.exe --help
```

---

## Advanced Setup (Alternative Methods)

### Option A: Using HuggingFace Transformers

Download and convert weights directly from HuggingFace (requires PyTorch):

```bash
cd 110-GPT2-hyungkyu
python utils/download_gpt2_weights.py
```

**Required packages:** `pip install torch transformers numpy`

### Option B: Manual Conversion

If you already have OpenAI checkpoint files:

```bash
cd 110-GPT2-hyungkyu
python utils/convert_openai_weights.py assets/weights/124M assets/weights/124M
```

**Required packages:** `pip install tensorflow numpy`

---

## Usage

### Text Generation

The program accepts command-line arguments for flexible text generation:

```bash
# Show help
./bin/debug/110-GPT2-hyungkyu.exe --help

# Use default settings (prompt: "The future of artificial intelligence is", max_tokens: 25)
./bin/debug/110-GPT2-hyungkyu.exe

# Custom prompt with default max_tokens (25)
./bin/debug/110-GPT2-hyungkyu.exe "Once upon a time"

# Custom prompt and token count
./bin/debug/110-GPT2-hyungkyu.exe "Once upon a time" 15

# Another example
./bin/debug/110-GPT2-hyungkyu.exe "Hello world" 20
```

**Command-line Arguments:**
- **Argument 1 (prompt)**: Text prompt for generation
  - Default: `"The future of artificial intelligence is"`
  - Example: `"Once upon a time"`
- **Argument 2 (max_tokens)**: Maximum number of tokens to generate
  - Default: `25`
  - Range: `1-100`
  - Example: `15`

**Help Flag:**
```bash
./bin/debug/110-GPT2-hyungkyu.exe --help
# or
./bin/debug/110-GPT2-hyungkyu.exe -h
```

### Generation Parameters

You can modify generation parameters in `model/gpt2Test.cpp`:

```cpp
runPromptGeneration(gpt2Net, tokenizer,
    prompt,
    max_tokens,
    0.0f,  // temperature: 0 = greedy (deterministic), >0 = sampling
    50,    // top_k: limit sampling to top-k tokens
    42     // seed: random seed for reproducibility
);
```

## Example Output

```
Configuration:
  Prompt: "The future of artificial intelligence is"
  Max tokens: 25

========================================
GPT-2 Text Generation Test (Pretrained Weights)
========================================

Loading configuration from: 110-GPT2-hyungkyu/assets/weights/124M/gpt2_config.txt
  Original config: vocab_size=50257, d_model=768, num_heads=12, num_layers=12
✓ Configuration loaded

Creating GPT-2 network...
✓ Network created

Loading pretrained weights from: 110-GPT2-hyungkyu/assets/weights/124M/gpt2_weights.bin
✓ All weights loaded successfully

=== Greedy Decoding (Deterministic) ===
Prompt: "The future of artificial intelligence is"
Max tokens: 25

--- Generated Text ---
The future of artificial intelligence is uncertain.

"We're not sure what the future will look like," said Dr. Michael S. Schoenfeld
--- End of Generation ---

Generated 25 new tokens (total: 31 tokens)
Generation time: 1250 ms (1.25 sec)
Generation speed: 20.00 tokens/sec
```

## Performance

- **Model**: GPT-2 Small (124M parameters)
- **Generation Speed**: ~20 tokens/sec (GPU dependent)
- **GPU Memory**: ~2GB for full model
- **Safe Token Limit**: 25 tokens per generation (due to buffer pool accumulation)

## Known Limitations

- **GPU Memory**: BufferPool accumulates memory across generations. Current safe limit is ~25 tokens per run.
- **Multiple Prompts**: Running multiple prompts in a single execution may cause GPU OOM. Run separately if needed.

## Troubleshooting

### "Failed to open vocab.json"
- Ensure `vocab.json` and `merges.txt` are in `assets/` folder
- Check file permissions

### "Pretrained weights not found"
- Run weight download/conversion scripts in `utils/`
- Verify `assets/weights/124M/gpt2_weights.bin` exists

### "VkResult is UNKNOWN_ERROR"
- GPU out of memory
- Reduce `max_tokens` parameter
- Close other GPU-intensive applications
- Try running only one test at a time

### Build Errors
- Ensure Vulkan SDK is installed
- Check CMake version (3.15+)
- Verify C++17 compiler support

## Architecture Details

### Model Configuration (GPT-2 Small)
- Vocabulary size: 50,257
- Context length: 1,024 tokens
- Hidden size (d_model): 768
- Number of layers: 12
- Number of attention heads: 12
- Feedforward size: 3,072 (4 × d_model)

### Components
- **Embedding Layer**: Token + positional embeddings
- **Transformer Blocks**: Multi-head attention + feedforward + layer normalization
- **Language Model Head**: Linear projection to vocabulary (weight-tied with token embeddings)

## References

- [OpenAI GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-2 on HuggingFace](https://huggingface.co/gpt2)
- [OpenAI GPT-2 Repository](https://github.com/openai/gpt-2)

## License

This is an educational implementation. GPT-2 weights are released by OpenAI under the MIT License.
