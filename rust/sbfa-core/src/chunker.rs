use pyo3::prelude::*;

/// Fast text chunking with overlap.
/// Python fallback: sbfa.rag.ingestion.chunk_text
#[pyfunction]
#[pyo3(signature = (text, chunk_size=512, overlap=64))]
pub fn fast_chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let bytes = text.as_bytes();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < bytes.len() {
        let end = (start + chunk_size).min(bytes.len());

        // Ensure we don't split in the middle of a UTF-8 character
        let end = if end < bytes.len() {
            let mut e = end;
            while e > start && !text.is_char_boundary(e) {
                e -= 1;
            }
            e
        } else {
            end
        };

        let chunk = &text[start..end];
        if !chunk.trim().is_empty() {
            chunks.push(chunk.to_string());
        }

        if end >= bytes.len() {
            break;
        }

        start = if end > overlap { end - overlap } else { 0 };
        // Ensure start is on a valid UTF-8 character boundary
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        assert!(fast_chunk_text("", 512, 64).is_empty());
    }

    #[test]
    fn test_short_text() {
        let result = fast_chunk_text("hello world", 512, 64);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "hello world");
    }

    #[test]
    fn test_chunking_with_overlap() {
        let text = "a".repeat(1024);
        let chunks = fast_chunk_text(&text, 512, 64);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_non_ascii_text() {
        // Japanese text: each character is 3 bytes in UTF-8
        let text = "あいうえおかきくけこさしすせそたちつてと";
        let chunks = fast_chunk_text(text, 10, 3);
        assert!(!chunks.is_empty());
        // Verify all chunks are valid UTF-8 (no panics)
        for chunk in &chunks {
            assert!(chunk.len() > 0);
        }
    }

    #[test]
    fn test_mixed_ascii_multibyte() {
        let text = "hello世界fooバー";
        let chunks = fast_chunk_text(text, 8, 2);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() > 0);
        }
    }
}
