//! Mimic voice engine: RVC v2 inference on ONNX Runtime with Applio-compatible
//! realtime block processing.

pub mod dsp;
pub mod gpu;
pub mod probe;
pub mod models;
pub mod pipeline;
pub mod prepare;
pub mod realtime;
pub mod wav;

pub use models::{Device, Models, Paths};
pub use pipeline::{BlockProcessor, BlockStats, Layout, Params};
pub use realtime::{DeviceInfo, LiveConfig, LiveSession, Telemetry, list_devices};
pub use realtime::{input_name, record_input, record_while};
