use embeddings::Embeddings;
use llm::LLM;
use vocab::Vocab;

mod llm;
mod embeddings;
mod vocab;
mod transformer;
mod feed_forward;
mod self_attention;
mod output_projection;
mod adam;

// Use the constants from lib.rs
const MAX_SEQ_LEN: usize = 40;
const EMBEDDING_DIM: usize = 4;
const HIDDEN_DIM: usize = 4;

fn main() {
    // Mock input
    let string = String::from("hello world </s>");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();
    
    // Add end of sequence token
    vocab_set.insert("</s>".to_string());
    
    let training_data = vec![
        ("hi how are you </s>", "I'm doing well, how about you? </s>"),
        ("hello how are </s>", "I'm great! How's your day? </s>"),
        ("hi there friend </s>", "Hey! It's nice to see you. </s>"),
        ("good morning how </s>", "Good morning! How's your day so far? </s>"),
        ("what is up </s>", "Not much, just here to chat! What about you? </s>"),
        ("how is it </s>", "It's going well! Thanks for asking. </s>"),
        ("nice to see </s>", "Nice to see you too! How have you been? </s>"),
        ("hi how is </s>", "Hi! How is your day going? </s>"),
        ("good evening friend </s>", "Good evening! Hope you had a great day. </s>"),
        ("what are you </s>", "I'm an AI here to chat with you! What's on your mind? </s>")
    ];
    
    // Process all training examples
    for (input, output) in &training_data {
        // Add words from inputs
        for word in input.split_whitespace() {
            vocab_set.insert(word.to_string());
        }
        
        // Add words from outputs
        for word in output.split_whitespace() {
            // Handle punctuation by splitting it from words
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        vocab_set.insert(current);
                        current = String::new();
                    }
                    vocab_set.insert(c.to_string());
                } else {
                    current.push(c);
                }
            }
            if !current.is_empty() {
                vocab_set.insert(current);
            }
        }
    }
    
    let vocab_words: Vec<String> = vocab_set.into_iter().collect();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let llm = LLM::new(vocab);

    llm.train(training_data, 100, 0.01);

    let result = llm.predict(&string);
    println!("output of LLM: {}", result);
}
