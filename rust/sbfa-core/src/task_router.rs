use pyo3::prelude::*;
use std::collections::HashMap;

/// Compute skill matching scores for agents against task keywords.
///
/// Args:
///     task_keywords: list of keywords extracted from the task
///     agent_tags: dict mapping agent_name -> list of skill tags
///
/// Returns:
///     list of (agent_name, score) tuples sorted by score descending
#[pyfunction]
pub fn compute_skill_scores(
    task_keywords: Vec<String>,
    agent_tags: HashMap<String, Vec<String>>,
) -> Vec<(String, f64)> {
    if task_keywords.is_empty() {
        return agent_tags
            .keys()
            .map(|name| (name.clone(), 0.0))
            .collect();
    }

    let task_kw_lower: Vec<String> = task_keywords.iter().map(|k| k.to_lowercase()).collect();

    let mut scores: Vec<(String, f64)> = agent_tags
        .iter()
        .map(|(name, tags)| {
            let tag_lower: Vec<String> = tags.iter().map(|t| t.to_lowercase()).collect();
            let matches = task_kw_lower
                .iter()
                .filter(|kw| tag_lower.iter().any(|tag| tag.contains(kw.as_str())))
                .count();
            let score = matches as f64 / task_kw_lower.len() as f64;
            (name.clone(), score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_scoring() {
        let keywords = vec!["code".to_string(), "debug".to_string()];
        let mut agent_tags = HashMap::new();
        agent_tags.insert(
            "claude".to_string(),
            vec!["code".to_string(), "debug".to_string(), "refactor".to_string()],
        );
        agent_tags.insert(
            "gemini".to_string(),
            vec!["multimodal".to_string(), "vision".to_string()],
        );

        let scores = compute_skill_scores(keywords, agent_tags);
        assert_eq!(scores[0].0, "claude");
        assert!(scores[0].1 > scores[1].1);
    }
}
