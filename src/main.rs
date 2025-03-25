use embeddings::Embeddings;
use output_projection::OutputProjection;
use transformer::TransformerBlock;
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
const EMBEDDING_DIM: usize = 400;
const HIDDEN_DIM: usize = 400;

fn main() {
    // Mock input
    let string = String::from("hi how are you </s>");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();
    
    // Add end of sequence token
    vocab_set.insert("</s>".to_string());
    
    let training_data = vec![
        ("hi how are you! I'm doing well, how about you? </s>"),
        ("hello how are? I'm great! How's your day? </s>"),
        ("hi there friend. Hey! It's nice to see you. </s>"),
        ("good morning how. Good morning! How's your day so far? </s>"),
        ("what is up Not much, just here to chat! What about you? </s>"),
        ("how is it It's going well! Thanks for asking. </s>"),
        ("nice to see Nice to see you too! How have you been? </s>"),
        ("hi how is Hi! How is your day going? </s>"),
        ("good evening friend Good evening! Hope you had a great day. </s>"),
        ("what are you I'm an AI here to chat with you! What's on your mind? </s>")
    ];
    
    // Process all training examples
    for (row) in &training_data {
        // Add words from outputs
        for word in row.split_whitespace() {
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

    let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    let mut llm = LLM::new(vocab, vec![
        Box::new(embeddings),
        Box::new(transformer_block),
        Box::new(output_projection),
    ]);

    println!("Before Training: {}", llm.predict(&string));
    llm.train(training_data, 50, 0.05);

    let result = llm.predict(&string);
    println!("After Training: {}", result);
}
