use pyo3::prelude::*;

mod chunker;
mod similarity;
mod task_router;

#[pymodule]
fn sbfa_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunker::fast_chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(task_router::compute_skill_scores, m)?)?;
    Ok(())
}
