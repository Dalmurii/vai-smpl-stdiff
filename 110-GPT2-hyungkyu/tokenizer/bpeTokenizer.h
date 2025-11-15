#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>

// TODO: Future improvements for production-ready tokenizer:
// 1. Add special token support (<|endoftext|>, <|startoftext|>, etc.)
// 2. Implement BPE caching for performance optimization
// 3. Use exact GPT-2 regex pattern (requires PCRE2 or similar library)
// 4. Add proper error handling and validation

class BPETokenizer {
private:
    // token string → id
    std::unordered_map<std::string, int> encoder;
    // id → token string
    std::unordered_map<int, std::string> decoder;
    // BPE merge rules: (pair) → rank
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // TODO: Add BPE cache for performance
    // std::unordered_map<std::string, std::vector<std::string>> bpe_cache;
    
    // Byte encoder/decoder for handling UTF-8
    std::unordered_map<unsigned char, std::string> byte_encoder;
    std::unordered_map<std::string, unsigned char> byte_decoder;
    
    void loadVocab(const std::string& vocab_file);
    void loadMerges(const std::string& merges_file);
    void initByteEncoder();
    
    std::string bytes_to_unicode(int b);
    std::vector<std::string> bpe(const std::string& token);
    std::set<std::pair<size_t, size_t>> get_pairs(const std::vector<std::string>& word);
    
public:
    BPETokenizer(const std::string& vocab_file, const std::string& merges_file);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);
    
    size_t vocab_size() const { return encoder.size(); }
};

#endif // BPE_TOKENIZER_H

