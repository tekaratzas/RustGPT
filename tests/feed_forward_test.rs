use llm::feed_forward::FeedForward;
use llm::{EMBEDDING_DIM, HIDDEN_DIM, Layer};
use ndarray::Array2;

#[test]
fn test_feed_forward_forward() {
    // Create feed-forward module
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = feed_forward.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_feed_forward_with_different_sequence_lengths() {
    // Create feed-forward module
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // Test forward pass
        let output = feed_forward.forward(&input);

        // Check output shape
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_feed_forward_and_backward() {
    // Create feed-forward module
    let mut feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = feed_forward.forward(&input);

    let grads = Array2::ones((3, HIDDEN_DIM));

    // Test backward pass
    let grad_input = feed_forward.backward(&grads, 0.01);

    // Make sure backward pass modifies the input
    assert_ne!(output, grad_input);
}
