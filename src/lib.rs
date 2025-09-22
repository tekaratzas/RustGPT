pub mod adam;
pub mod embeddings;
pub mod feed_forward;
pub mod layer_norm;
pub mod llm;
pub mod output_projection;
pub mod self_attention;
pub mod transformer;
pub mod vocab;
// Re-export key structs for easier access
pub use embeddings::Embeddings;
pub use llm::LLM;
pub use llm::Layer;
pub use vocab::Vocab;

// Constants
pub const MAX_SEQ_LEN: usize = 40;
pub const EMBEDDING_DIM: usize = 32;
pub const HIDDEN_DIM: usize = 32;
