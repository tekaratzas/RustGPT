use llm::{Layer, EMBEDDING_DIM};
use ndarray::Array2;
use llm::output_projection::OutputProjection;

#[test]
fn test_output_projection_creation() {
    let vocab_size = 10;
    let output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);
    
    // Check weight matrix dimensions
    assert_eq!(output_proj.w_out.shape(), [EMBEDDING_DIM, vocab_size]);
    
    // Check bias vector dimensions
    assert_eq!(output_proj.b_out.shape(), [1, vocab_size]);
    
    // Check optimizer dimensions
    assert_eq!(output_proj.optimizer.m.shape(), [EMBEDDING_DIM, vocab_size]);
    assert_eq!(output_proj.optimizer.v.shape(), [EMBEDDING_DIM, vocab_size]);
}

#[test]
fn test_output_projection_forward() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);
    
    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));
    
    // Test forward pass
    let output = output_proj.forward(&input);
    
    // Check output shape - should be [seq_len, vocab_size]
    assert_eq!(output.shape(), [3, vocab_size]);
}

#[test]
fn test_output_projection_with_different_sequence_lengths() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);
    
    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        
        // Test forward pass
        let output = output_proj.forward(&input);
        
        // Check output shape
        assert_eq!(output.shape(), [seq_len, vocab_size]);
    }
}

#[test]
fn test_output_projection_backward() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);
    
    // Create input tensor
    let input = Array2::ones((3, EMBEDDING_DIM));
    
    // Forward pass first (required to cache input)
    let _output = output_proj.forward(&input);
    
    // Create gradient tensor
    let grads = Array2::ones((3, vocab_size));
    
    // Test backward pass
    let grad_input = output_proj.backward(&grads, 0.01);
    
    // Check gradient input shape
    assert_eq!(grad_input.shape(), [3, EMBEDDING_DIM]);
    
    // Verify that parameters were updated
    let w_out_before = output_proj.w_out.clone();
    let b_out_before = output_proj.b_out.clone();
    
    // Run another forward and backward pass
    let _output = output_proj.forward(&input);
    let _grad_input = output_proj.backward(&grads, 0.01);
    
    // Check that parameters changed
    assert_ne!(output_proj.w_out, w_out_before);
    assert_ne!(output_proj.b_out, b_out_before);
}

#[test]
fn test_output_projection_training() {
    let vocab_size = 10;
    let mut output_proj = OutputProjection::new(EMBEDDING_DIM, vocab_size);
    
    // Create input tensor
    let input = Array2::ones((3, EMBEDDING_DIM));
    
    // Run multiple training steps
    for _ in 0..5 {
        // Forward pass
        let _output = output_proj.forward(&input);
        
        // Create gradient tensor (simulating cross-entropy loss gradients)
        let mut grads = Array2::zeros((3, vocab_size));
        grads[[0, 0]] = 1.0; // Set gradient for first token
        
        // Backward pass
        let _grad_input = output_proj.backward(&grads, 0.01);
    }
    
    // Verify that parameters were updated
    assert_ne!(output_proj.w_out.sum(), 0.0);
    assert_ne!(output_proj.b_out.sum(), 0.0);
} 