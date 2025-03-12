use llm::Vocab;

#[test]
fn test_vocab_encode_decode() {
    let words = vec!["hello", "world", "this", "is", "rust", "</s>"];
    let vocab = Vocab::new(words);
    
    // Test encoding
    assert_eq!(vocab.encode("hello"), Some(0));
    assert_eq!(vocab.encode("world"), Some(1));
    assert_eq!(vocab.encode("unknown"), None);
    
    // Test decoding
    assert_eq!(vocab.decode(0).map(|s| s.as_str()), Some("hello"));
    assert_eq!(vocab.decode(1).map(|s| s.as_str()), Some("world"));
    assert_eq!(vocab.decode(999), None);
}

#[test]
fn test_vocab_default() {
    let vocab = Vocab::default();
    
    // Test that default vocab contains expected words
    assert!(vocab.encode("hello").is_some());
    assert!(vocab.encode("world").is_some());
    assert!(vocab.encode("</s>").is_some());
} 