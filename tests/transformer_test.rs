use llm::transformer::TransformerBlock;
use llm::{EMBEDDING_DIM, HIDDEN_DIM, Layer};
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
