use llm::{EMBEDDING_DIM, Embeddings, Layer, MAX_SEQ_LEN, Vocab};

#[test]
fn test_embeddings_creation() {
    // Create with custom vocab
    let words = vec!["hello", "world", "test", "</s>"];
    let _vocab = Vocab::new(words); // Fix unused variable warning
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

#[test]
fn test_embedding_backwards() {
    // Create vocab and embeddings
    let vocab = Vocab::default();
    let mut embeddings = Embeddings::new(vocab);

    let pre_train_token_embeddings = embeddings.token_embeddings.clone();
    let pre_train_position_embeddings = embeddings.positional_embeddings.clone();

    // Simulate forward and backward pass
    use ndarray::Array2;
    let input = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 2.0]).unwrap();
    let _output = embeddings.forward(&input);

    // Create some dummy gradients and run backward pass
    let grads = Array2::from_shape_vec((3, EMBEDDING_DIM), vec![0.1; 3 * EMBEDDING_DIM]).unwrap();
    let _grad_input = embeddings.backward(&grads, 0.01);

    let post_train_token_embeddings = embeddings.token_embeddings.clone();
    let post_train_position_embeddings = embeddings.positional_embeddings.clone();

    assert_ne!(pre_train_token_embeddings, post_train_token_embeddings);
    assert_ne!(
        pre_train_position_embeddings,
        post_train_position_embeddings
    );
}
