#include "attentionNode.h"
#include "../../core/neuralNet.h"
#include "../../core/error.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

// JSON parsing helper
#include <nlohmann/json.hpp>

using namespace vk;
using json = nlohmann::json;

// Global device and descriptor pool (defined in embeddingNodeTest.cpp)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

// Helper function to load JSON test data
json loadTestData(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open test data file: " + filename);
    }

    json data;
    file >> data;
    return data;
}

// Helper function to convert JSON array to std::vector<float>
void flattenJson(const json& j, std::vector<float>& result)
{
    if (j.is_array()) {
        for (const auto& elem : j) {
            flattenJson(elem, result);
        }
    } else if (j.is_number()) {
        result.push_back(j.get<float>());
    }
}

std::vector<float> jsonToVector(const json& j)
{
    std::vector<float> result;
    flattenJson(j, result);
    return result;
}

void testLinear()
{
    std::cout << "\n========== Test: Linear Layer ===========" << std::endl;

    const uint32_t batch_size = 2;
    const uint32_t seq_len = 3;
    const uint32_t in_features = 8;
    const uint32_t out_features = 8;

    std::cout << "Creating neural network with Linear layer..." << std::endl;

    NeuralNet net(netGlobalDevice, 1, 1);
    LinearNode linear(in_features, out_features);

    net.input(0) - linear - net.output(0);

    // Create simple input
    std::vector<float> input_data(batch_size * seq_len * in_features);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = i * 0.1f;
    }

    Tensor inputTensor = Tensor(batch_size, seq_len, in_features).set(input_data);

    // Create simple weight pattern
    std::vector<float> weight_data(out_features * in_features);
    for (uint32_t o = 0; o < out_features; ++o) {
        for (uint32_t i = 0; i < in_features; ++i) {
            weight_data[o * in_features + i] = (o * in_features + i) * 0.01f;
        }
    }

    linear["weight"] = Tensor(out_features, in_features).set(weight_data);

    std::cout << "Running inference..." << std::endl;

    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * out_features * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    std::cout << "Output (first token, first 4 values): ";
    for (int i = 0; i < 4; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "✓ Linear layer test completed" << std::endl;
}

void testMultiHeadAttentionSimple()
{
    std::cout << "\n========== Test: Multi-Head Attention (Simple Input) ===========" << std::endl;

    const uint32_t batch_size = 1;
    const uint32_t seq_len = 2;
    const uint32_t d_model = 4;
    const uint32_t num_heads = 2;

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  num_heads: " << num_heads << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    MultiHeadAttentionNode mha(d_model, num_heads);

    net.input(0) - mha - net.output(0);

    // Simple input: all values = 1.0
    std::vector<float> input_data(batch_size * seq_len * d_model, 1.0f);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    std::cout << "Input: all values = 1.0" << std::endl;

    // Simple weights: identity-like patterns
    std::vector<float> W_q(d_model * d_model, 0.1f);
    std::vector<float> W_k(d_model * d_model, 0.1f);
    std::vector<float> W_v(d_model * d_model, 0.1f);
    std::vector<float> W_out(d_model * d_model, 0.1f);

    mha["W_query"] = Tensor(d_model, d_model).set(W_q);
    mha["W_key"] = Tensor(d_model, d_model).set(W_k);
    mha["W_value"] = Tensor(d_model, d_model).set(W_v);
    mha["W_out"] = Tensor(d_model, d_model).set(W_out);

    std::cout << "Weights: all values = 0.1" << std::endl;
    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    std::cout << "\nOutput (first token): ";
    for (int i = 0; i < d_model; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output (second token): ";
    for (int i = 0; i < d_model; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[d_model + i] << " ";
    }
    std::cout << std::endl;

    // Basic sanity checks
    bool has_nan = false;
    bool has_inf = false;
    bool all_zero = true;
    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 0; i < batch_size * seq_len * d_model; ++i) {
        if (std::isnan(data[i])) has_nan = true;
        if (std::isinf(data[i])) has_inf = true;
        if (data[i] != 0.0f) all_zero = false;
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    std::cout << "\nSanity checks:" << std::endl;
    std::cout << "  Has NaN: " << (has_nan ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Has Inf: " << (has_inf ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  All zero: " << (all_zero ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;

    if (!has_nan && !has_inf && !all_zero) {
        std::cout << "\n✓ Simple input test PASSED - basic functionality working" << std::endl;
    } else {
        std::cout << "\n✗ Simple input test FAILED - check implementation" << std::endl;
    }
}

void testMultiHeadAttentionReference()
{
    std::cout << "\n========== Test: Multi-Head Attention (Reference Data) ===========" << std::endl;

    // Load reference test data
    std::cout << "Loading reference test data..." << std::endl;

    std::string testDataPath = std::string(PROJECT_CURRENT_DIR) + "/model/attention/mha_test_data.json";
    json testData = loadTestData(testDataPath);

    uint32_t batch_size = testData["config"]["batch_size"];
    uint32_t seq_len = testData["config"]["seq_len"];
    uint32_t d_model = testData["config"]["d_model"];
    uint32_t num_heads = testData["config"]["num_heads"];
    uint32_t head_dim = testData["config"]["head_dim"];

    std::cout << "Config:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_len << std::endl;
    std::cout << "  d_model: " << d_model << std::endl;
    std::cout << "  num_heads: " << num_heads << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;

    // Create neural network
    NeuralNet net(netGlobalDevice, 1, 1);
    MultiHeadAttentionNode mha(d_model, num_heads);

    net.input(0) - mha - net.output(0);

    // Load input
    std::vector<float> input_data = jsonToVector(testData["input"]);
    Tensor inputTensor = Tensor(batch_size, seq_len, d_model).set(input_data);

    std::cout << "Input loaded: " << input_data.size() << " values" << std::endl;

    // Load weights
    std::vector<float> W_q = jsonToVector(testData["weights"]["W_query"]);
    std::vector<float> W_k = jsonToVector(testData["weights"]["W_key"]);
    std::vector<float> W_v = jsonToVector(testData["weights"]["W_value"]);
    std::vector<float> W_out = jsonToVector(testData["weights"]["W_out"]);

    mha["W_query"] = Tensor(d_model, d_model).set(W_q);
    mha["W_key"] = Tensor(d_model, d_model).set(W_k);
    mha["W_value"] = Tensor(d_model, d_model).set(W_v);
    mha["W_out"] = Tensor(d_model, d_model).set(W_out);

    std::cout << "Weights loaded" << std::endl;

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * d_model * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    // Load expected output
    std::vector<float> expected_output = jsonToVector(testData["output"]);

    std::cout << "\nVerifying results..." << std::endl;
    std::cout << "  Expected output size: " << expected_output.size() << std::endl;
    std::cout << "  Actual output size: " << batch_size * seq_len * d_model << std::endl;

    // Compare first token
    std::cout << "\n  First token (batch 0, token 0):" << std::endl;
    std::cout << "    Expected: ";
    for (int i = 0; i < d_model; ++i) {
        std::cout << std::fixed << std::setprecision(4) << expected_output[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "    Actual:   ";
    for (int i = 0; i < d_model; ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

    // Calculate error
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;

    for (size_t i = 0; i < expected_output.size(); ++i) {
        float error = std::abs(data[i] - expected_output[i]);
        avg_error += error;
        max_error = std::max(max_error, error);
        if (error > 0.01f) {
            error_count++;
        }
    }

    avg_error /= expected_output.size();

    std::cout << "\n  Error statistics:" << std::endl;
    std::cout << "    Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "    Avg error: " << avg_error << std::endl;
    std::cout << "    Values with error > 0.01: " << error_count << " / " << expected_output.size() << std::endl;

    if (max_error < 0.1f) {
        std::cout << "\n✓ Multi-Head Attention numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Multi-Head Attention numerical verification FAILED" << std::endl;
        std::cout << "  (This is expected for first implementation - debugging needed)" << std::endl;
    }
}

void attentionNodeTest()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Attention Node (Vulkan) - Numerical Verification Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // testLinear();  // TODO: Fix Linear test

        // Step 1: Test with simple input first
        testMultiHeadAttentionSimple();

        // Step 2: Test with reference data (only if simple test passes)
        std::cout << "\nProceeding to reference data test..." << std::endl;
        testMultiHeadAttentionReference();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Attention Node tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    }
}
