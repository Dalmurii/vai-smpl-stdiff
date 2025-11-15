"""
Generate reference test data for Multi-Head Attention using NumPy
This creates known inputs and expected outputs for numerical verification
"""

import numpy as np
import json


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def linear(x, weight):
    """Linear transformation: x @ weight.T"""
    return x @ weight.T


def multi_head_attention(x, W_q, W_k, W_v, W_out, num_heads):
    """
    Multi-head attention implementation

    Args:
        x: (batch, seq_len, d_model)
        W_q, W_k, W_v: (d_model, d_model) - weight matrices
        W_out: (d_model, d_model) - output projection
        num_heads: number of attention heads

    Returns:
        output: (batch, seq_len, d_model)
        intermediates: dictionary with intermediate results
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    # Project to Q, K, V
    Q = linear(x, W_q)  # (batch, seq_len, d_model)
    K = linear(x, W_k)
    V = linear(x, W_v)

    # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
    Q = Q.reshape(batch, seq_len, num_heads, head_dim)
    K = K.reshape(batch, seq_len, num_heads, head_dim)
    V = V.reshape(batch, seq_len, num_heads, head_dim)

    # Transpose: (batch, num_heads, seq_len, head_dim)
    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    # (batch, num_heads, seq_len, seq_len)
    attn_scores = Q @ K.transpose(0, 1, 3, 2)

    # Scale
    attn_scores = attn_scores / np.sqrt(head_dim)

    # Apply causal mask (set upper triangle to -inf)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    attn_scores[:, :, mask] = -1e38  # Use large negative number instead of -inf for JSON compatibility

    # Softmax
    attn_weights = softmax(attn_scores, axis=-1)

    # Weighted sum
    context = attn_weights @ V  # (batch, num_heads, seq_len, head_dim)

    # Combine heads: (batch, seq_len, num_heads, head_dim)
    context = context.transpose(0, 2, 1, 3)

    # Reshape: (batch, seq_len, d_model)
    context = context.reshape(batch, seq_len, d_model)

    # Output projection
    output = linear(context, W_out)

    intermediates = {
        'Q': Q,
        'K': K,
        'V': V,
        'attn_scores': attn_scores,
        'attn_weights': attn_weights,
        'context': context
    }

    return output, intermediates


def generate_test_case():
    """Generate test case with small dimensions for easy verification"""

    # Small dimensions for manual verification
    batch_size = 2
    seq_len = 3
    d_model = 8
    num_heads = 2
    head_dim = d_model // num_heads  # 4

    print("=" * 60)
    print("Generating Multi-Head Attention Test Case")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"d_model: {d_model}")
    print(f"num_heads: {num_heads}")
    print(f"head_dim: {head_dim}")
    print()

    # Set seed for reproducibility
    np.random.seed(42)

    # Initialize weights with simple patterns
    W_q = np.zeros((d_model, d_model))
    W_k = np.zeros((d_model, d_model))
    W_v = np.zeros((d_model, d_model))
    W_out = np.zeros((d_model, d_model))

    # W_query: pattern based on indices
    for i in range(d_model):
        for j in range(d_model):
            W_q[i, j] = (i * d_model + j) * 0.01

    # W_key: slightly different pattern
    for i in range(d_model):
        for j in range(d_model):
            W_k[i, j] = (i * d_model + j) * 0.01 + 0.1

    # W_value: another pattern
    for i in range(d_model):
        for j in range(d_model):
            W_v[i, j] = (i * d_model + j) * 0.01 + 0.2

    # Output projection
    for i in range(d_model):
        for j in range(d_model):
            W_out[i, j] = (i * d_model + j) * 0.01 + 0.3

    # Create input with simple pattern
    x = np.zeros((batch_size, seq_len, d_model))
    for b in range(batch_size):
        for s in range(seq_len):
            for d in range(d_model):
                x[b, s, d] = (b * 100 + s * 10 + d) * 0.1

    print("Input shape:", x.shape)
    print("Input (first batch, first token):", x[0, 0].tolist())
    print()

    # Forward pass
    output, intermediates = multi_head_attention(x, W_q, W_k, W_v, W_out, num_heads)

    # Save all data
    test_data = {
        'config': {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model,
            'num_heads': num_heads,
            'head_dim': head_dim
        },
        'input': x.tolist(),
        'weights': {
            'W_query': W_q.tolist(),
            'W_key': W_k.tolist(),
            'W_value': W_v.tolist(),
            'W_out': W_out.tolist()
        },
        'intermediates': {
            'Q': intermediates['Q'].tolist(),
            'K': intermediates['K'].tolist(),
            'V': intermediates['V'].tolist(),
            'attn_scores': intermediates['attn_scores'].tolist(),
            'attn_weights': intermediates['attn_weights'].tolist(),
            'context': intermediates['context'].tolist()
        },
        'output': output.tolist()
    }

    # Save to JSON
    with open('mha_test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print("Test data saved to 'mha_test_data.json'")
    print()
    print("Output shape:", output.shape)
    print("Output (first batch, first token):", output[0, 0].tolist())
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Attention weights sum (should be ~1.0): {intermediates['attn_weights'][0, 0, 0].sum():.6f}")
    print()

    # Print first attention weights for verification
    print("First batch, first head, attention weights:")
    print(intermediates['attn_weights'][0, 0])
    print()

    return test_data


if __name__ == "__main__":
    test_data = generate_test_case()
    print("Reference test case generation complete!")
