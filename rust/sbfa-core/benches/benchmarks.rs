use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

// Note: These benchmarks test the Rust functions directly (not via PyO3).

fn bench_chunk_text(c: &mut Criterion) {
    let text = "a".repeat(100_000);
    c.bench_function("chunk_text_100k", |b| {
        b.iter(|| sbfa_core::fast_chunk_text(&text, 512, 64))
    });
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let a: Vec<f32> = (0..1536).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..1536).map(|i| (i as f32).cos()).collect();
    c.bench_function("cosine_similarity_1536d", |b_iter| {
        b_iter.iter(|| sbfa_core::cosine_similarity(a.clone(), b.clone()))
    });
}

fn bench_skill_scores(c: &mut Criterion) {
    let keywords: Vec<String> = vec!["code", "debug", "refactor"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut agent_tags = HashMap::new();
    for i in 0..10 {
        agent_tags.insert(
            format!("agent_{i}"),
            vec!["code", "debug", "vision", "general"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
    }
    c.bench_function("skill_scores_10_agents", |b| {
        b.iter(|| sbfa_core::compute_skill_scores(keywords.clone(), agent_tags.clone()))
    });
}

criterion_group!(benches, bench_chunk_text, bench_cosine_similarity, bench_skill_scores);
criterion_main!(benches);
