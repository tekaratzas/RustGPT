use llm::{Embeddings, Vocab, EMBEDDING_DIM, MAX_SEQ_LEN};
use ndarray::Array2;

#[test]
fn test_embeddings_creation() {
    // Test default embeddings creation
    let embeddings = Embeddings::default();
    
    // Create with custom vocab
    let words = vec!["hello", "world", "test", "</s>"];
    let vocab = Vocab::new(words);
    let custom_embeddings = Embeddings::new(vocab);
}

#[test]
fn test_embed_tokens() {
    // Create vocab and embeddings
    let words = vec!["hello", "world", "test", "</s>"];
    let vocab = Vocab::new(words);
    let embeddings = Embeddings::new(vocab.clone());
    
    // Test embedding a single token
    let token_ids = vec![0]; // "hello"
    let embedded = embeddings.embed_tokens(&token_ids);
    
    // Check dimensions
    assert_eq!(embedded.shape(), [1, EMBEDDING_DIM]);
    
    // Test embedding multiple tokens
    let token_ids = vec![0, 1, 2]; // "hello world test"
    let embedded = embeddings.embed_tokens(&token_ids);
    
    // Check dimensions
    assert_eq!(embedded.shape(), [3, EMBEDDING_DIM]);
}

#[test]
fn test_positional_embeddings() {
    // Create vocab and embeddings
    let words = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
    let vocab = Vocab::new(words);
    let embeddings = Embeddings::new(vocab);
    
    // Test with different sequence lengths
    for seq_len in 1..5 {
        let token_ids = vec![0; seq_len]; // Repeat token 0 seq_len times
        let embedded = embeddings.embed_tokens(&token_ids);
        
        // Check dimensions
        assert_eq!(embedded.shape(), [seq_len, EMBEDDING_DIM]);
        
        // Verify that embeddings for the same token at different positions are different
        // (due to positional embeddings being added)
        if seq_len > 1 {
            let first_pos = embedded.row(0).to_owned();
            let second_pos = embedded.row(1).to_owned();
            
            // They should be different due to positional encoding
            assert_ne!(first_pos, second_pos);
        }
    }
}

#[test]
fn test_max_sequence_length() {
    // Create vocab and embeddings
    let vocab = Vocab::default();
    let embeddings = Embeddings::new(vocab);
    
    // Create a sequence at the maximum length
    let token_ids = vec![0; MAX_SEQ_LEN];
    let embedded = embeddings.embed_tokens(&token_ids);
    
    // Check dimensions
    assert_eq!(embedded.shape(), [MAX_SEQ_LEN, EMBEDDING_DIM]);
} 