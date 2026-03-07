use pyo3::prelude::*;

/// Compute cosine similarity between two vectors.
#[pyfunction]
pub fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute cosine similarity between a query vector and multiple candidate vectors.
/// Returns a list of (index, score) tuples sorted by score descending.
#[pyfunction]
#[pyo3(signature = (query, candidates, top_k=5))]
pub fn batch_cosine_similarity(
    query: Vec<f32>,
    candidates: Vec<Vec<f32>>,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let mut scores: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(query.clone(), c.clone())))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(v.clone(), v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(a, b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_batch() {
        let query = vec![1.0, 0.0];
        let candidates = vec![
            vec![1.0, 0.0], // most similar
            vec![0.0, 1.0], // least similar
            vec![0.7, 0.7], // middle
        ];
        let results = batch_cosine_similarity(query, candidates, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // index of most similar
    }
}
