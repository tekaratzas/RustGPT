use llm::{BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, BatchLayer, BatchFeedForward};
use ndarray::Array3;
use ndarray::Array2;

#[test]
fn test_feed_forward_forward() {
    // Create feed-forward module
    let mut feed_forward = BatchFeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array3::ones((BATCH_SIZE, 3, EMBEDDING_DIM));

    // Test forward pass
    let output = feed_forward.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_feed_forward_with_different_sequence_lengths() {
    // Create feed-forward module
    let mut feed_forward = BatchFeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array3::ones((BATCH_SIZE, seq_len, EMBEDDING_DIM));

        // Test forward pass
        let output = feed_forward.forward(&input);

        // Check output shape
        assert_eq!(output.shape(), [BATCH_SIZE, seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_feed_forward_and_backward() {
    // Create feed-forward module
    let mut feed_forward = BatchFeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array3::ones((BATCH_SIZE, 3, EMBEDDING_DIM));

    // Test forward pass
    let output = feed_forward.forward(&input);

    let grads = Array3::ones((BATCH_SIZE, 3, EMBEDDING_DIM));

    // Test backward pass
    let grad_input = feed_forward.backward(&grads, 0.01);

    // Make sure backward pass modifies the input
    assert_ne!(output, grad_input);
}
