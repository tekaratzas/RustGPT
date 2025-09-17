pub mod llm;
pub mod embeddings;
pub mod vocab;
pub mod transformer;
pub mod feed_forward;
pub mod self_attention;
pub mod output_projection;
pub mod layer_norm;
pub mod optimizer;
pub mod optimizers;

// Re-export key structs for easier access
pub use vocab::Vocab;
pub use embeddings::Embeddings;
pub use llm::LLM;
pub use llm::Layer;

// Constants
pub const MAX_SEQ_LEN: usize = 40;
pub const EMBEDDING_DIM: usize = 32;
pub const HIDDEN_DIM: usize = 32;