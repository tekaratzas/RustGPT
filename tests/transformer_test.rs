use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, Layer,
    transformer::{MultiHeadTransformerBlock, TransformerBlock},
};
use ndarray::Array2;

#[test]
fn test_transformer_block() {
    let mut transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Create a simple input tensor
    let input = Array2::ones((1, EMBEDDING_DIM));

    // Test forward pass
    let output = transformer.forward(&input);

    // Check output shape
    assert_eq!(output.shape(), [1, EMBEDDING_DIM]);
}

#[test]
fn test_multi_head_transformer_block_forward() {
    let num_heads = 8;
    let mut transformer = MultiHeadTransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, num_heads);

    // Create some input
    let input = Array2::ones((5, EMBEDDING_DIM));

    let output = transformer.forward(&input);

    // Check output shape
    assert_eq!(output.shape(), [5, EMBEDDING_DIM]);
}

#[test]
fn test_multi_head_transformer_block_backward() {
    let num_heads = 4;
    let mut transformer = MultiHeadTransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, num_heads);

    let input = Array2::ones((3, EMBEDDING_DIM));
    let _output = transformer.forward(&input);

    // Backward pass
    let grad_output = Array2::ones((3, EMBEDDING_DIM));
    let grad_input = transformer.backward(&grad_output, 0.001);

    // Check gradient shape
    assert_eq!(grad_input.shape(), input.shape());
}

#[test]
fn test_multi_head_transformer_block_parameter_count() {
    let num_heads = 8;
    let transformer = MultiHeadTransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, num_heads);

    // Should have more parameters than the single-head version due to W_o matrix
    let params = transformer.parameters();
    assert!(params > 0, "Should have non-zero parameters");
}

#[test]
fn test_multi_head_transformer_different_sequence_lengths() {
    let num_heads = 8;
    let mut transformer = MultiHeadTransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, num_heads);

    // Test with different sequence lengths
    for seq_len in 1..10 {
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        let output = transformer.forward(&input);
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_multi_head_transformer_layer_type() {
    let num_heads = 8;
    let transformer = MultiHeadTransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, num_heads);
    assert_eq!(transformer.layer_type(), "MultiHeadTransformerBlock");
}
