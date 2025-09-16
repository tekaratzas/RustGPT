use llm::{Layer, EMBEDDING_DIM, HIDDEN_DIM};
use ndarray::Array2;
use llm::transformer::TransformerBlock;
use llm::optimizer::OptimizerType;

#[test]
fn test_transformer_block() {
    let optimizer_choice = OptimizerType::AdamW { weight_decay: 0.01 };
    let mut transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM, &optimizer_choice);
    
    // Create a simple input tensor
    let input = Array2::ones((1, EMBEDDING_DIM));
    
    // Test forward pass
    let output = transformer.forward(&input);
    
    // Check output shape
    assert_eq!(output.shape(), [1, EMBEDDING_DIM]);
}