//! Text chunking and sentence boundary splitting for long chapters.

/// Tách một văn bản dài (chương truyện) thành các câu ngắn hoàn chỉnh.
pub fn split_sentences(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        current.push(c);

        let is_punct = matches!(c, '.' | '!' | '?' | '…');
        let is_newline = c == '\n';

        if is_punct {
            // Kiểm tra nếu là dấu chấm giữa các con số (e.g. 3.14)
            let is_decimal = if c == '.' && i > 0 && i + 1 < len {
                chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit()
            } else {
                false
            };

            if !is_decimal {
                // Kiểm tra xem ký tự tiếp theo có phải khoảng trắng, newline hoặc hết chuỗi
                let next_is_space = i + 1 == len || chars[i + 1].is_whitespace();
                if next_is_space {
                    let clean = clean_sentence(&current);
                    if !clean.is_empty() {
                        sentences.push(clean);
                    }
                    current.clear();
                }
            }
        } else if is_newline {
            // Nếu gặp 2 dòng xuống liên tiếp hoặc kết thúc câu
            let next_is_newline = i + 1 < len && chars[i + 1] == '\n';
            if next_is_newline || current.trim().ends_with(['.', '!', '?', ';', ':']) {
                let clean = clean_sentence(&current);
                if !clean.is_empty() {
                    sentences.push(clean);
                }
                current.clear();
            }
        }

        i += 1;
    }

    let clean = clean_sentence(&current);
    if !clean.is_empty() {
        sentences.push(clean);
    }

    // Nếu có câu nào quá dài (> 220 ký tự), chia tiếp theo dấu phẩy
    let mut final_sentences = Vec::new();
    for s in sentences {
        if s.chars().count() > 220 {
            final_sentences.extend(split_by_commas(&s, 160));
        } else {
            final_sentences.push(s);
        }
    }

    final_sentences
}

/// Làm sạch câu, loại bỏ các ký tự thừa ở đầu/cuối
fn clean_sentence(s: &str) -> String {
    let mut trimmed = s.trim().to_string();
    trimmed = trimmed.replace(';', ",");
    
    // Thu gọn khoảng trắng thừa
    let mut clean_words = Vec::new();
    for w in trimmed.split_whitespace() {
        clean_words.push(w);
    }
    clean_words.join(" ")
}

/// Tách các vế câu theo dấu phẩy khi câu quá dài
fn split_by_commas(s: &str, max_chars: usize) -> Vec<String> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();

    if parts.len() <= 1 {
        return vec![s.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for (idx, part) in parts.iter().enumerate() {
        let is_last = idx == parts.len() - 1;
        let suffix = if is_last { "" } else { "," };
        let piece = format!("{}{}", part, suffix);

        if current.is_empty() {
            current = piece;
        } else if current.chars().count() + piece.chars().count() + 1 <= max_chars {
            current.push(' ');
            current.push_str(&piece);
        } else {
            chunks.push(current);
            current = piece;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences() {
        let text = "Tôi là một siêu cấp lão làng. Có một tựa game di động nọ, tên của nó là Battle of Summoners!\n\nPhổ biến không? Hầu như là không.";
        let sents = split_sentences(text);
        assert_eq!(sents.len(), 4);
        assert_eq!(sents[0], "Tôi là một siêu cấp lão làng.");
        assert_eq!(sents[1], "Có một tựa game di động nọ, tên của nó là Battle of Summoners!");
        assert_eq!(sents[2], "Phổ biến không?");
        assert_eq!(sents[3], "Hầu như là không.");
    }
}
