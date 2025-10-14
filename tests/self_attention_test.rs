use llm::{Layer, EMBEDDING_DIM};
use ndarray::Array2;
use llm::self_attention::SelfAttention;

// #[test]
// fn test_self_attention_forward() {
//     // Create self-attention module
//     let mut self_attention = SelfAttention::new(EMBEDDING_DIM);
    
//     // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
//     let input = Array2::ones((3, EMBEDDING_DIM));
    
//     // Test forward pass
//     let output = self_attention.forward(&input);
    
//     // Check output shape - should be same as input
//     assert_eq!(output.shape(), input.shape());
// }

// #[test]
// fn test_self_attention_with_different_sequence_lengths() {
//     // Create self-attention module
//     let mut self_attention = SelfAttention::new(EMBEDDING_DIM);
    
//     // Test with different sequence lengths
//     for seq_len in 1..5 {
//         // Create input tensor
//         let input = Array2::ones((seq_len, EMBEDDING_DIM));
        
//         // Test forward pass
//         let output = self_attention.forward(&input);
        
//         // Check output shape
//         assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
//     }
// } 



#[test]
fn test_multi_head_attention_forward() {
    // Create multi-head self-attention module with 8 heads
    let num_heads = 8;
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM, num_heads);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = self_attention.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_multi_head_attention_with_different_sequence_lengths() {
    // Create multi-head self-attention module with 4 heads
    let num_heads = 4;
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM, num_heads);

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

#[test]
fn test_multi_head_attention_different_head_counts() {
    // Test with different numbers of heads (must divide embedding_dim evenly)
    let valid_head_counts = vec![1, 2, 4, 8, 16, 32, 64, 128];
    
    for num_heads in valid_head_counts {
        let mut self_attention = SelfAttention::new(EMBEDDING_DIM, num_heads);
        let input = Array2::ones((3, EMBEDDING_DIM));
        let output = self_attention.forward(&input);
        
        // Verify output shape is correct
        assert_eq!(output.shape(), [3, EMBEDDING_DIM]);
        
        // Verify parameters are calculated correctly
        // Q, K, V, O projection matrices: 4 * (EMBEDDING_DIM * EMBEDDING_DIM)
        let expected_params = 4 * EMBEDDING_DIM * EMBEDDING_DIM;
        assert_eq!(self_attention.parameters(), expected_params);
    }
}

#[test]
fn test_multi_head_attention_backward() {
    let num_heads = 8;
    let mut self_attention = SelfAttention::new(EMBEDDING_DIM, num_heads);
    
    // Forward pass
    let input = Array2::from_elem((4, EMBEDDING_DIM), 0.5);
    let _output = self_attention.forward(&input);
    
    // Backward pass with gradient
    let grad_output = Array2::ones((4, EMBEDDING_DIM));
    let grad_input = self_attention.backward(&grad_output, 0.01);
    
    // Check gradient shape matches input shape
    assert_eq!(grad_input.shape(), input.shape());
}
