# Rust LLM Project

A simple implementation of a language model in Rust.

## Project Structure

- `src/`: Source code
  - `lib.rs`: Library exports
  - `main.rs`: Main entry point
  - `llm.rs`: LLM implementation
  - `transformer.rs`: Transformer block implementation
  - `vocab.rs`: Vocabulary handling
  - `embeddings.rs`: Token embedding implementation
  - `feed_forward.rs`: Feed-forward network implementation
  - `self_attention.rs`: Self-attention mechanism implementation
  - `output_projection.rs`: Output projection layer

- `tests/`: Test files
  - `vocab_test.rs`: Tests for vocabulary functionality
  - `transformer_test.rs`: Tests for transformer block
  - `llm_test.rs`: Tests for the LLM implementation
  - `embeddings_test.rs`: Tests for the embeddings component
  - `self_attention_test.rs`: Tests for the self-attention mechanism
  - `feed_forward_test.rs`: Tests for the feed-forward network

## Running the Project

```bash
# Run the main application
cargo run

# Build the project
cargo build
```

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific component
cargo test --test vocab_test
cargo test --test transformer_test
cargo test --test llm_test
cargo test --test embeddings_test
cargo test --test self_attention_test
cargo test --test feed_forward_test

# Run tests with output
cargo test -- --nocapture
``` 