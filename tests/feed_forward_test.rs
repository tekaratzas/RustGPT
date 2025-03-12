use llm::{Layer, EMBEDDING_DIM, HIDDEN_DIM};
use ndarray::Array2;
use llm::feed_forward::FeedForward;

#[test]
fn test_feed_forward_creation() {
    // Create feed-forward module
    let feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);
}

#[test]
fn test_feed_forward_forward() {
    // Create feed-forward module
    let feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);
    
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
    let feed_forward = FeedForward::new(EMBEDDING_DIM, HIDDEN_DIM);
    
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