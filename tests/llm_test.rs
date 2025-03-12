use llm::{LLM, Vocab};

#[test]
fn test_llm_tokenize() {
    let vocab = Vocab::default();
    let llm = LLM::new(vocab);
    
    // Test tokenization
    let tokens = llm.tokenize("hello world");
    assert!(!tokens.is_empty());
    
    // Test that tokens can be decoded back
    for token in tokens {
        assert!(llm.vocab.decode(token).is_some());
    }
}

#[test]
fn test_llm_predict() {
    let vocab = Vocab::default();
    let llm = LLM::new(vocab);
    
    // Test prediction
    let result = llm.predict("hello world");
    assert!(!result.is_empty());
} 