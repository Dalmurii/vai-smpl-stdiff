#ifndef ATTENTION_NODE_H
#define ATTENTION_NODE_H

#include "../../core/neuralNet.h"
#include "../../core/vulkanApp.h"

using namespace vk;

// Global device and descriptor pool (defined in test file)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

/**
 * Linear transformation: Y = X @ W^T
 * Input: [batch, seq_len, in_features]
 * Weight: [out_features, in_features]
 * Output: [batch, seq_len, out_features]
 */
class LinearNode : public Node
{
    uint32_t in_features;
    uint32_t out_features;

    ComputePipeline linearPipeline;
    DescriptorSet linearDescSet;

public:
    LinearNode(uint32_t in_features, uint32_t out_features);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Softmax along the last dimension
 * Input: [batch, ..., dim]
 * Output: [batch, ..., dim] (sum along last dim = 1.0)
 */
class SoftmaxNode : public Node
{
    ComputePipeline softmaxPipeline;
    DescriptorSet softmaxDescSet;

public:
    SoftmaxNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Multi-Head Attention
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Internal weights:
 * - W_query: [d_model, d_model]
 * - W_key: [d_model, d_model]
 * - W_value: [d_model, d_model]
 * - W_out: [d_model, d_model]
 */
class MultiHeadAttentionNode : public Node
{
    uint32_t d_model;
    uint32_t num_heads;
    uint32_t head_dim;

    // Pipelines for each stage
    ComputePipeline qkvProjection;        // Project input to Q, K, V
    ComputePipeline reshapeForHeads;      // Reshape to multi-head format
    ComputePipeline attentionScores;      // Q @ K^T / sqrt(head_dim)
    ComputePipeline applyCausalMask;      // Set upper triangle to -inf
    ComputePipeline softmaxPipeline;      // Softmax on attention scores
    ComputePipeline weightedSum;          // attn_weights @ V
    ComputePipeline combineHeads;         // Reshape and concat heads
    ComputePipeline outputProjection;     // Final linear projection

    // Descriptor sets
    DescriptorSet qkvProjDescSet;
    DescriptorSet reshapeDescSet;
    DescriptorSet scoresDescSet;
    DescriptorSet maskDescSet;
    DescriptorSet softmaxDescSet;
    DescriptorSet weightedSumDescSet;
    DescriptorSet combineDescSet;
    DescriptorSet outProjDescSet;

public:
    MultiHeadAttentionNode(uint32_t d_model, uint32_t num_heads);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

#endif // ATTENTION_NODE_H
