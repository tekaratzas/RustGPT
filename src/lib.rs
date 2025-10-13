pub mod adam;
pub mod batch_mode;
pub mod dataset_loader;
pub mod embeddings;
pub mod feed_forward;
pub mod layer_norm;
pub mod llm;
pub mod output_projection;
pub mod self_attention;
pub mod transformer;
pub mod vocab;
// Re-export key structs for easier access
pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use llm::{LLM, Layer};
pub use vocab::Vocab;

// Re-export batch_mode structs
pub use batch_mode::llm::LLM as BatchLLM;
pub use batch_mode::llm::Layer as BatchLayer;
pub use batch_mode::output_projection::OutputProjection as BatchOutputProjection;
pub use batch_mode::feed_forward::FeedForward as BatchFeedForward;

// Constants
pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
pub const BATCH_SIZE: usize = 4;
