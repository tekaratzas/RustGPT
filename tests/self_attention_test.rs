use llm::{Layer, EMBEDDING_DIM};
use ndarray::Array2;
use llm::self_attention::SelfAttention;

#[test]
fn test_self_attention_creation() {
    // Create self-attention module
    let self_attention = SelfAttention::new(EMBEDDING_DIM);
}

#[test]
fn test_self_attention_forward() {
    // Create self-attention module
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);
    
    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));
    
    // Test forward pass
    let output = self_attention.forward(&input);
    
    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_self_attention_with_different_sequence_lengths() {
    // Create self-attention module
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM);
    
    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        
        // Test forward pass
        let output = self_attention.forward(&input);
        
        // Check output shape
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
} 