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
├── assets/
│   ├── vocab.json              # GPT-2 vocabulary (50,257 tokens)
│   ├── merges.txt              # BPE merge rules (50,000 rules)
│   ├── the-verdict.txt         # Test data file
│   └── weights/
│       └── 124M/               # GPT-2 model weights
│           ├── gpt2_weights.bin      # Binary weights (converted)
│           └── gpt2_config.txt       # Model configuration
├── utils/
│   ├── download_gpt2_weights.py      # Download weights from HuggingFace
│   └── convert_openai_weights.py     # Convert OpenAI checkpoint to binary
├── debug/
│   └── check_*.py              # Weight verification scripts
├── core/                       # Core neural network framework
├── tokenizer/                  # BPE tokenizer implementation
├── model/                      # GPT-2 model architecture
│   ├── gpt2Net.h              # Main GPT-2 network
│   ├── gpt2Weights.h          # Weight loading
│   ├── gpt2Generation.h       # Text generation
│   └── gpt2Test.cpp           # Test functions
└── main.cpp                    # Entry point

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

### Step 1: Download Tokenizer Files

The tokenizer files (`vocab.json` and `merges.txt`) should already be in the `assets/` folder. If not, download them:

```bash
cd 110-GPT2-hyungkyu/assets
curl -o vocab.json https://huggingface.co/gpt2/resolve/main/vocab.json
curl -o merges.txt https://huggingface.co/gpt2/resolve/main/merges.txt
```

### Step 2: Download and Convert Weights

#### Option A: From HuggingFace (One-Step)

Download and convert weights directly from HuggingFace:

```bash
cd 110-GPT2-hyungkyu
python utils/download_gpt2_weights.py
```

This will:
- Download GPT-2 weights from HuggingFace
- Convert to binary format automatically
- Save to `assets/weights/124M/gpt2_weights.bin`

#### Option B: From OpenAI Checkpoint (Two-Step)

If you have OpenAI's original TensorFlow checkpoint:

1. Download the checkpoint to `assets/weights/124M/`:
   ```bash
   # Download from https://openaipublic.blob.core.windows.net/gpt-2/models/124M/
   # Files needed: model.ckpt.*, hparams.json, checkpoint
   ```

2. Convert to binary format:
   ```bash
   cd 110-GPT2-hyungkyu
   python utils/convert_openai_weights.py assets/weights/124M assets/weights/124M
   ```

This will create:
- `assets/weights/124M/gpt2_weights.bin` - Binary weight file (~500MB)
- `assets/weights/124M/gpt2_config.txt` - Model configuration

### Step 3: Build the Project

```bash
# From vai-samples root directory
cmake --build build --config Debug --target 110-GPT2-hyungkyu
```

Or for Release build:
```bash
cmake --build build --config Release --target 110-GPT2-hyungkyu
```

### Step 4: Run Tests

```bash
# From vai-samples root directory
./bin/debug/110-GPT2-hyungkyu.exe
```

## Usage

### Text Generation

The main program (`main.cpp`) provides several testing options:

```cpp
int main()
{
    // Option 1: Run all basic tests (uses GPU memory)
    // runBasicTests();

    // Option 2: Run text generation with random weights (lightweight)
    // testGPT2Generation();

    // Option 3: Run pretrained weights test (requires more GPU memory)
    testGPT2Pretrained();  // Default: prompt="The future of artificial intelligence is", max_tokens=25

    // Custom usage examples:
    // testGPT2Pretrained("Once upon a time", 15);  // Custom prompt and token count
    // testGPT2Pretrained("Hello world");  // Custom prompt, default max_tokens=25

    return 0;
}
```

### Customizing Text Generation

Edit `main.cpp` to customize the generation:

```cpp
// Custom prompt and token count
std::string prompt = "The future of artificial intelligence is";
uint32_t max_tokens = 28;
testGPT2Pretrained(prompt, max_tokens);
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
