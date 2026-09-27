// =============================================================================
// Magna Middleware — Generic Postprocessor
// =============================================================================
//! Task-agnostic postprocessing for inference outputs.
//!
//! This module provides:
//! - **Generic** raw tensor output (for any model)
//! - **Classification** postprocessing (optional, when labels are loaded)
//!
//! The middleware no longer assumes the model is a classifier.

use std::fmt;
use tracing::debug;

use crate::inference::traits::TensorBuffer;
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

// ---------------------------------------------------------------------------
// Generic inference result
// ---------------------------------------------------------------------------

/// Generic inference result — task-agnostic.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    /// Generic list of generic output tensors.
    pub outputs: Vec<TensorBuffer>,
    /// Optional classification (populated only if labels are loaded).
    pub classification: Option<ClassificationResult>,
}

impl InferenceResult {
    /// Create from a list of generic output tensors, with optional classification.
    pub fn from_outputs(
        outputs: &[TensorBuffer],
        classifier: Option<&Classifier>,
    ) -> MiddlewareResult<Self> {
        let classification = if let Some(cls) = classifier {
            if (cls.has_labels() || cls.softmax) && !outputs.is_empty() {
                // Use the first output for classification (classic classification assumption)
                let f32_data = tensor_to_f32(&outputs[0])?;
                Some(cls.classify_from_scores(&f32_data)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            outputs: outputs.to_vec(),
            classification,
        })
    }
}

impl fmt::Display for InferenceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InferenceResult({} outputs)", self.outputs.len())?;
        if let Some(ref cls) = self.classification {
            write!(f, ", {}", cls)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Classification result (optional)
// ---------------------------------------------------------------------------

/// Classification result — only populated when labels are available.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationResult {
    pub top1_label: String,
    pub top1_score: f32,
    pub top1_index: usize,
    pub top5: Vec<(String, f32, usize)>,
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  Top-1: {} ({:.4}) [index {}]",
            self.top1_label, self.top1_score, self.top1_index
        )?;
        for (i, (label, score, idx)) in self.top5.iter().enumerate() {
            writeln!(
                f,
                "  Top-{}: {} ({:.4}) [index {}]",
                i + 1,
                label,
                score,
                idx
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Classifier (optional component)
// ---------------------------------------------------------------------------

/// Classifier for converting raw scores into human-readable predictions.
///
/// If no labels are loaded, returns numeric class indices.
pub struct Classifier {
    labels: Vec<String>,
    pub softmax: bool,
}

impl Classifier {
    /// Create a classifier with no labels.
    pub fn new(softmax: bool) -> Self {
        Self {
            labels: Vec::new(),
            softmax,
        }
    }

    /// Load class labels from a file (one label per line).
    pub fn load_labels_from_file(path: &str, softmax: bool) -> MiddlewareResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            MiddlewareError::ConfigError(format!("Failed to load labels from '{}': {}", path, e))
        })?;

        let labels: Vec<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();

        debug!(
            "[Classifier] Loaded {} class labels from '{}'",
            labels.len(),
            path
        );

        Ok(Self { labels, softmax })
    }

    /// Whether this classifier has labels loaded.
    pub fn has_labels(&self) -> bool {
        !self.labels.is_empty()
    }

    /// Classify from a pre-extracted f32 score array.
    pub fn classify_from_scores(&self, scores: &[f32]) -> MiddlewareResult<ClassificationResult> {
        if scores.is_empty() {
            return Err(MiddlewareError::PostprocessingFailed(
                "Empty output tensor".into(),
            ));
        }

        // Optionally apply softmax
        let probs = if self.softmax {
            softmax(scores)
        } else {
            scores.to_vec()
        };

        // Find top-5
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(5);

        let get_label = |idx: usize| -> String {
            if idx < self.labels.len() {
                self.labels[idx].clone()
            } else {
                format!("Class {}", idx)
            }
        };

        let top1 = indexed.first().ok_or_else(|| {
            MiddlewareError::PostprocessingFailed("No scores available to classify".into())
        })?;
        let top5: Vec<_> = indexed
            .iter()
            .map(|&(idx, score)| (get_label(idx), score, idx))
            .collect();

        Ok(ClassificationResult {
            top1_label: get_label(top1.0),
            top1_score: top1.1,
            top1_index: top1.0,
            top5,
        })
    }

    /// Classify a TensorBuffer output.
    pub fn classify(&self, output: &TensorBuffer) -> MiddlewareResult<ClassificationResult> {
        let scores = tensor_to_f32(output)?;
        self.classify_from_scores(&scores)
    }
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Classifier")
            .field("num_labels", &self.labels.len())
            .field("softmax", &self.softmax)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tensor_to_f32(tensor: &TensorBuffer) -> MiddlewareResult<Vec<f32>> {
    match tensor.precision {
        crate::utils::errors::Precision::FP32 => Ok(tensor.try_as_f32_slice()?.to_vec()),
        crate::utils::errors::Precision::FP16 => {
            let f16_slice: &[u16] = bytemuck::cast_slice(&tensor.data);
            Ok(f16_slice
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect())
        }
        other => Err(MiddlewareError::PostprocessingFailed(format!(
            "Cannot postprocess {:?} tensors directly — convert to FP32 first",
            other
        ))),
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::errors::Precision;

    #[test]
    fn classifier_no_labels() {
        let cls = Classifier::new(true);
        assert!(!cls.has_labels());

        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let result = cls.classify_from_scores(&scores).unwrap();
        assert_eq!(result.top1_index, 9); // Highest score
        assert!(result.top1_label.contains("Class 9"));
    }

    #[test]
    fn classifier_with_labels() {
        let cls = Classifier {
            labels: vec!["cat".into(), "dog".into(), "fish".into()],
            softmax: false,
        };
        assert!(cls.has_labels());

        let scores = vec![0.1f32, 0.9, 0.5];
        let result = cls.classify_from_scores(&scores).unwrap();
        assert_eq!(result.top1_label, "dog");
        assert_eq!(result.top1_index, 1);
    }

    #[test]
    fn softmax_normalization() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Largest logit should have largest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn generic_inference_result() {
        let output = TensorBuffer {
            name: "output".to_string(),
            data: vec![0u8; 40],
            shape: vec![1, 10],
            precision: Precision::FP32,
        };
        let result = InferenceResult::from_outputs(&[output], None).unwrap();
        assert_eq!(result.outputs[0].data.len(), 40);
        assert!(result.classification.is_none());
    }

    #[test]
    fn generic_with_classifier() {
        let output = TensorBuffer::from_f32("output", &[0.1, 0.5, 0.9], vec![1, 3]);
        let cls = Classifier::new(true);
        let result = InferenceResult::from_outputs(&[output], Some(&cls)).unwrap();
        assert!(result.classification.is_some());
        assert_eq!(result.classification.unwrap().top1_index, 2);
    }

    #[test]
    fn empty_scores_fails() {
        let cls = Classifier::new(false);
        assert!(cls.classify_from_scores(&[]).is_err());
    }
}
