# GPT-2 Tokenizer Files

## Required Files

This project requires two tokenizer files:

1. **vocab.json** - GPT-2 vocabulary (50,257 tokens)
2. **merges.txt** - BPE merge rules (50,000 rules)

## Download Instructions

### Option 1: From HuggingFace (Recommended)

Download directly from the GPT-2 model repository:

```bash
# vocab.json
curl -o vocab.json https://huggingface.co/gpt2/resolve/main/vocab.json

# merges.txt
curl -o merges.txt https://huggingface.co/gpt2/resolve/main/merges.txt
```

Or visit: https://huggingface.co/gpt2/tree/main

### Option 2: From tiktoken

The files are also available in the tiktoken repository format.

### File Placement

Place both files in the `110-GPT2-hyungkyu/` directory:
```
110-GPT2-hyungkyu/
  ├── vocab.json
  ├── merges.txt
  ├── main.cpp
  └── ...
```

## Verification

- **vocab.json**: Should be ~1MB, containing 50,257 token mappings
- **merges.txt**: Should be ~456KB, containing ~50,000 merge rules

## Current Status

✅ **Completed:**
- Project structure setup (CMakeLists.txt, error.h)
- Downloaded vocab.json and merges.txt from HuggingFace
- Implemented BPETokenizer class with encode/decode methods
- Successfully tokenizes text into token IDs
- Vocab size: 50,257 tokens
- BPE merge rules: 50,000 rules

⚠️ **Known Limitations:**
The current implementation successfully encodes text to token IDs, but the decode function needs refinement for perfect round-trip conversion. The byte-level encoding/decoding mechanism requires more sophisticated handling of UTF-8 characters and special tokens.

**Improvements Needed:**
1. Refine byte_encoder/decoder for proper UTF-8 handling
2. Implement GPT-2's exact regex pattern for text splitting
3. Add special token handling (<|endoftext|>, etc.)
4. Optimize BPE algorithm performance

**Test Results:**
- "Hello, world!" → Token IDs: [15496, 11, 6894, 0]
- Successfully generates token IDs for all test cases
- Basic tokenization working correctly

