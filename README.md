# 🦀 Rust LLM from Scratch

https://github.com/user-attachments/assets/ec4a4100-b03a-4b3c-a7d6-806ea54ed4ed

A complete **Large Language Model implementation in pure Rust** with no external ML frameworks. Built from the ground up using only `ndarray` for matrix operations.

## 🚀 What This Is

This project demonstrates how to build a transformer-based language model from scratch in Rust, including:
- **Pre-training** on factual text completion
- **Instruction tuning** for conversational AI
- **Interactive chat mode** for testing
- **Full backpropagation** with gradient clipping
- **Modular architecture** with clean separation of concerns

## 🔍 Key Files to Explore

Start with these two core files to understand the implementation:

- **[`src/main.rs`](src/main.rs)** - Training pipeline, data preparation, and interactive mode
- **[`src/llm.rs`](src/llm.rs)** - Core LLM implementation with forward/backward passes and training logic

## 🏗️ Architecture

The model uses a **transformer-based architecture** with the following components:

```
Input Text → Tokenization → Embeddings → Transformer Blocks → Output Projection → Predictions
```

### Project Structure

```
src/
├── main.rs              # 🎯 Training pipeline and interactive mode
├── llm.rs               # 🧠 Core LLM implementation and training logic
├── lib.rs               # 📚 Library exports and constants
├── transformer.rs       # 🔄 Transformer block (attention + feed-forward)
├── self_attention.rs    # 👀 Multi-head self-attention mechanism  
├── feed_forward.rs      # ⚡ Position-wise feed-forward networks
├── embeddings.rs        # 📊 Token embedding layer
├── output_projection.rs # 🎰 Final linear layer for vocabulary predictions
├── vocab.rs            # 📝 Vocabulary management and tokenization
├── layer_norm.rs       # 🧮 Layer normalization
└── adam.rs             # 🏃 Adam optimizer implementation

tests/
├── llm_test.rs         # Tests for core LLM functionality
├── transformer_test.rs # Tests for transformer blocks
├── self_attention_test.rs # Tests for attention mechanisms
├── feed_forward_test.rs # Tests for feed-forward layers
├── embeddings_test.rs  # Tests for embedding layers
├── vocab_test.rs       # Tests for vocabulary handling
├── adam_test.rs        # Tests for optimizer
└── output_projection_test.rs # Tests for output layer
```

## 🧪 What The Model Learns

The implementation includes two training phases:

1. **Pre-training**: Learns basic world knowledge from factual statements
   - "The sun rises in the east and sets in the west"
   - "Water flows downhill due to gravity"
   - "Mountains are tall and rocky formations"

2. **Instruction Tuning**: Learns conversational patterns
   - "User: How do mountains form? Assistant: Mountains are formed through tectonic forces..."
   - Handles greetings, explanations, and follow-up questions

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/tekaratzas/RustGPT.git 
cd RustGPT
cargo run

# The model will:
# 1. Build vocabulary from training data
# 2. Pre-train on factual statements (100 epochs)  
# 3. Instruction-tune on conversational data (100 epochs)
# 4. Enter interactive mode for testing
```

## 🎮 Interactive Mode

After training, test the model interactively:

```
Enter prompt: How do mountains form?
Model output: Mountains are formed through tectonic forces or volcanism over long geological time periods

Enter prompt: What causes rain?
Model output: Rain is caused by water vapor in clouds condensing into droplets that become too heavy to remain airborne
```

## 🧮 Technical Implementation

### Model Configuration
- **Vocabulary Size**: Dynamic (built from training data)
- **Embedding Dimension**: 128
- **Hidden Dimension**: 256  
- **Max Sequence Length**: 80 tokens
- **Architecture**: 3 Transformer blocks + embeddings + output projection

### Training Details
- **Optimizer**: Adam with gradient clipping
- **Pre-training LR**: 0.0005 (100 epochs)
- **Instruction Tuning LR**: 0.0001 (100 epochs)
- **Loss Function**: Cross-entropy loss
- **Gradient Clipping**: L2 norm capped at 5.0

### Key Features
- **Custom tokenization** with punctuation handling
- **Greedy decoding** for text generation
- **Gradient clipping** for training stability
- **Modular layer system** with clean interfaces
- **Comprehensive test coverage** for all components

## 🔧 Development

```bash
# Run all tests
cargo test

# Test specific components
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test

# Build optimized version
cargo build --release

# Run with verbose output
cargo test -- --nocapture
```

## 🧠 Learning Resources

This implementation demonstrates key ML concepts:
- **Transformer architecture** (attention, feed-forward, layer norm)
- **Backpropagation** through neural networks
- **Language model training** (pre-training + fine-tuning)
- **Tokenization** and vocabulary management
- **Gradient-based optimization** with Adam

Perfect for understanding how modern LLMs work under the hood!

## 📊 Dependencies

- `ndarray` - N-dimensional arrays for matrix operations
- `rand` + `rand_distr` - Random number generation for initialization

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!

## 🤝 Contributing

Contributions are welcome! This project is perfect for learning and experimentation.

### High Priority Features Needed
- **🏪 Model Persistence** - Save/load trained parameters to disk (currently all in-memory)
- **⚡ Performance optimizations** - SIMD, parallel training, memory efficiency
- **🎯 Better sampling** - Beam search, top-k/top-p, temperature scaling
- **📊 Evaluation metrics** - Perplexity, benchmarks, training visualizations

### Areas for Improvement
- **Advanced architectures** (multi-head attention, positional encoding, RoPE)
- **Training improvements** (different optimizers, learning rate schedules, regularization)
- **Data handling** (larger datasets, tokenizer improvements, streaming)
- **Model analysis** (attention visualization, gradient analysis, interpretability)

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/model-persistence`
3. Make your changes and add tests
4. Run the test suite: `cargo test`
5. Submit a pull request with a clear description

### Code Style
- Follow standard Rust conventions (`cargo fmt`)
- Add comprehensive tests for new features
- Update documentation and README as needed
- Keep the "from scratch" philosophy - avoid heavy ML dependencies

### Ideas for Contributions
- 🚀 **Beginner**: Model save/load, more training data, config files
- 🔥 **Intermediate**: Beam search, positional encodings, training checkpoints
- ⚡ **Advanced**: Multi-head attention, layer parallelization, custom optimizations

Questions? Open an issue or start a discussion! 

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra! 
