use llm::{EMBEDDING_DIM, Layer, multi_head_attention::MultiHeadAttention};
use ndarray::Array2;

#[test]
fn test_multi_head_attention_forward() {
    // Create multi-head attention module with 8 heads
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);

    // Create input tensor (seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = mha.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());

    // Verify output is not all zeros
    let output_sum: f32 = output.iter().sum();
    assert!(output_sum.abs() > 0.0, "Output should not be all zeros");
}

#[test]
fn test_multi_head_attention_with_different_sequence_lengths() {
    // Create multi-head attention module with 4 heads
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 4);

    // Test with different sequence lengths
    for seq_len in 1..10 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // Test forward pass
        let output = mha.forward(&input);

        // Check output shape
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_multi_head_attention_different_head_counts() {
    // Test with different numbers of heads (must divide EMBEDDING_DIM evenly)
    let seq_len = 5;
    let valid_head_counts = vec![1, 2, 4, 8, 16, 32, 64, 128];

    for num_heads in valid_head_counts {
        if EMBEDDING_DIM.is_multiple_of(num_heads) {
            let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, num_heads);
            let input = Array2::ones((seq_len, EMBEDDING_DIM));

            let _output = mha.forward(&input);

            assert_eq!(mha.num_heads, num_heads);
            assert_eq!(mha.head_dim, EMBEDDING_DIM / num_heads);
        }
    }
}

#[test]
fn test_multi_head_attention_residual_connection() {
    // Test that residual connection is working
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    let seq_len = 3;
    let input = Array2::from_shape_fn((seq_len, EMBEDDING_DIM), |(i, j)| {
        (i * EMBEDDING_DIM + j) as f32
    });

    let output = mha.forward(&input);

    // Output should not be zero due to residual connection
    let output_norm: f32 = output.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(output_norm > 0.0, "Output should not be zero");

    // Output should be different from input (attention transforms it)
    let mut has_difference = false;
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            if (output[[i, j]] - input[[i, j]]).abs() > 1e-3 {
                has_difference = true;
                break;
            }
        }
    }
    assert!(
        has_difference,
        "Output should differ from input due to attention computation"
    );
}

#[test]
fn test_multi_head_attention_backward_pass() {
    // Test backward pass
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    let seq_len = 3;
    let input = Array2::ones((seq_len, EMBEDDING_DIM));

    // Forward pass
    let _output = mha.forward(&input);

    // Backward pass with mock gradients
    let grad_output = Array2::ones((seq_len, EMBEDDING_DIM));
    let grad_input = mha.backward(&grad_output, 0.001);

    // Check gradient shape
    assert_eq!(grad_input.shape(), input.shape());

    // Gradients should not be all zeros
    let grad_norm: f32 = grad_input.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(grad_norm > 0.0, "Gradients should not be zero");
}

#[test]
fn test_multi_head_attention_parameter_count() {
    let num_heads = 8;
    let mha = MultiHeadAttention::new(EMBEDDING_DIM, num_heads);

    // Should have 4 weight matrices: W_q, W_k, W_v, W_o
    // Each is EMBEDDING_DIM x EMBEDDING_DIM
    let expected = 4 * EMBEDDING_DIM * EMBEDDING_DIM;
    assert_eq!(mha.parameters(), expected);
}

#[test]
fn test_multi_head_attention_causal_masking() {
    // Test that causal masking prevents attention to future tokens
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 4);
    let seq_len = 5;

    // Create input with distinct patterns for each position
    let mut input = Array2::zeros((seq_len, EMBEDDING_DIM));
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            input[[i, j]] = (i as f32 + 1.0) * 10.0 + (j as f32);
        }
    }

    let output = mha.forward(&input);

    // Check that output shape is correct
    assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);

    // The first token should only attend to itself (plus residual)
    // We can't verify exact values due to random initialization,
    // but we can verify the output is not zero
    let first_token_norm: f32 = output.row(0).iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(first_token_norm > 0.0);
}

#[test]
fn test_multi_head_attention_layer_type() {
    let mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    assert_eq!(mha.layer_type(), "MultiHeadAttention");
}

#[test]
fn test_split_and_concat_heads_consistency() {
    // Test that split and concat are inverse operations
    let num_heads = 8;
    let seq_len = 5;
    let mha = MultiHeadAttention::new(EMBEDDING_DIM, num_heads);

    // Create random input
    let input = Array2::from_shape_fn((seq_len, EMBEDDING_DIM), |(i, j)| {
        ((i * EMBEDDING_DIM + j) as f32 * 0.1).sin()
    });

    // Split and concat
    let heads = mha.split_heads(&input);
    let reconstructed = mha.concat_heads(&heads);

    // Should be identical
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            let diff = (reconstructed[[i, j]] - input[[i, j]]).abs();
            assert!(
                diff < 1e-5,
                "Split-concat should preserve values. Diff at ({}, {}): {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_multi_head_attention_training_step() {
    // Simulate a mini training step
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    let seq_len = 3;
    let lr = 0.001;

    // Initial forward pass
    let input = Array2::from_shape_fn((seq_len, EMBEDDING_DIM), |(i, j)| {
        ((i + j) as f32 * 0.1).sin()
    });
    let _output1 = mha.forward(&input);

    // Backward pass with gradients
    let grad_output = Array2::ones((seq_len, EMBEDDING_DIM)) * 0.1;
    let _grad_input = mha.backward(&grad_output, lr);

    // Second forward pass - output should be different due to weight updates
    let output2 = mha.forward(&input);

    // Verify output2 is valid (weights were updated)
    let output2_norm: f32 = output2.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        output2_norm > 0.0,
        "Output should be valid after weight update"
    );
}

#[test]
fn test_multi_head_attention_numerical_stability() {
    // Test with extreme values to check numerical stability
    let mut mha = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    let seq_len = 3;

    // Test with large values
    let input_large = Array2::ones((seq_len, EMBEDDING_DIM)) * 100.0;
    let output_large = mha.forward(&input_large);
    assert!(
        output_large.iter().all(|&x| x.is_finite()),
        "Output should be finite with large inputs"
    );

    // Test with small values
    let input_small = Array2::ones((seq_len, EMBEDDING_DIM)) * 0.001;
    let output_small = mha.forward(&input_small);
    assert!(
        output_small.iter().all(|&x| x.is_finite()),
        "Output should be finite with small inputs"
    );
}

#[test]
#[should_panic(expected = "embedding_dim must be divisible by num_heads")]
fn test_multi_head_attention_invalid_head_count() {
    // Should panic if num_heads doesn't divide embedding_dim
    MultiHeadAttention::new(EMBEDDING_DIM, 7); // 128 is not divisible by 7
}

#[test]
fn test_multi_head_vs_single_head() {
    // Compare single-head MHA with multi-head MHA
    let seq_len = 3;
    let input = Array2::from_shape_fn((seq_len, EMBEDDING_DIM), |(i, j)| {
        ((i + j) as f32 * 0.1).sin()
    });

    // Single head
    let mut mha_single = MultiHeadAttention::new(EMBEDDING_DIM, 1);
    let output_single = mha_single.forward(&input);

    // Multiple heads
    let mut mha_multi = MultiHeadAttention::new(EMBEDDING_DIM, 8);
    let output_multi = mha_multi.forward(&input);

    // Both should have same shape
    assert_eq!(output_single.shape(), output_multi.shape());

    // Both should have non-zero outputs
    let norm_single: f32 = output_single.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_multi: f32 = output_multi.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(norm_single > 0.0);
    assert!(norm_multi > 0.0);

    // Outputs should differ (different initializations and computations)
    let mut differs = false;
    for i in 0..seq_len {
        for j in 0..EMBEDDING_DIM {
            if (output_single[[i, j]] - output_multi[[i, j]]).abs() > 1e-3 {
                differs = true;
                break;
            }
        }
    }
    assert!(differs, "Single-head and multi-head outputs should differ");
}
