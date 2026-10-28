//! Vietnamese text normalization for ZeroTTS: expands numbers, percentages, dates into spoken Vietnamese words.

use regex::Regex;

/// Chuyển đổi số nguyên (tối đa hàng nghìn tỷ) sang chữ tiếng Việt.
pub fn int_to_vietnamese_words(n: u64) -> String {
    if n == 0 {
        return "không".to_string();
    }

    let digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"];
    let scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"];

    // Phân tách số thành từng nhóm 3 chữ số từ phải sang trái
    let mut num = n;
    let mut groups = Vec::new();
    while num > 0 {
        groups.push((num % 1000) as u16);
        num /= 1000;
    }

    let mut result_parts: Vec<String> = Vec::new();
    let num_groups = groups.len();

    for (i, &chunk) in groups.iter().enumerate().rev() {
        if chunk == 0 {
            continue;
        }

        let is_highest = i == num_groups - 1;
        let c_hundred = (chunk / 100) as usize;
        let c_ten = ((chunk % 100) / 10) as usize;
        let c_unit = (chunk % 10) as usize;

        let mut chunk_words = Vec::new();

        // Xử lý hàng trăm
        if c_hundred > 0 || !is_highest {
            chunk_words.push(format!("{} trăm", digits[c_hundred]));
        }

        // Xử lý hàng chục & đơn vị
        if c_ten > 1 {
            chunk_words.push(format!("{} mươi", digits[c_ten]));
            if c_unit == 1 {
                chunk_words.push("mốt".to_string());
            } else if c_unit == 4 {
                chunk_words.push("tư".to_string());
            } else if c_unit == 5 {
                chunk_words.push("lăm".to_string());
            } else if c_unit > 0 {
                chunk_words.push(digits[c_unit].to_string());
            }
        } else if c_ten == 1 {
            chunk_words.push("mười".to_string());
            if c_unit == 5 {
                chunk_words.push("lăm".to_string());
            } else if c_unit > 0 {
                chunk_words.push(digits[c_unit].to_string());
            }
        } else {
            // c_ten == 0
            if c_hundred > 0 || !is_highest {
                if c_unit > 0 {
                    chunk_words.push("linh".to_string());
                    chunk_words.push(digits[c_unit].to_string());
                }
            } else if c_unit > 0 {
                chunk_words.push(digits[c_unit].to_string());
            }
        }

        let scale = scales[i];
        if !scale.is_empty() {
            chunk_words.push(scale.to_string());
        }

        result_parts.push(chunk_words.join(" "));
    }

    result_parts.join(" ")
}

/// Chuẩn hóa chuỗi văn bản tiếng Việt:
/// 1. Số phần trăm (100% -> một trăm phần trăm)
/// 2. Ngày tháng (27/10/2026 -> ngày 27 tháng 10 năm 2026 -> ngày hai mươi bảy...)
/// 3. Số nguyên độc lập (100 -> một trăm, 24 -> hai mươi tư, ...)
pub fn normalize_vietnamese_text(text: &str) -> String {
    let mut s = text.to_string();

    // 1. Chuẩn hóa ngày tháng dạng dd/mm/yyyy hoặc d/m/yyyy (kèm hoặc không kèm từ "ngày" phía trước)
    let re_date = Regex::new(r"(?i)(?:ngày\s+)?\b(\d{1,2})/(\d{1,2})/(\d{4})\b").unwrap();
    s = re_date
        .replace_all(&s, |caps: &regex::Captures| {
            let day: u64 = caps[1].parse().unwrap_or(0);
            let month: u64 = caps[2].parse().unwrap_or(0);
            let year: u64 = caps[3].parse().unwrap_or(0);
            let day_str = int_to_vietnamese_words(day);
            let month_str = if month == 4 { "tư".to_string() } else { int_to_vietnamese_words(month) };
            let year_str = int_to_vietnamese_words(year);
            format!("ngày {} tháng {} năm {}", day_str, month_str, year_str)
        })
        .to_string();

    // 2. Chuẩn hóa phần trăm: 100% -> một trăm phần trăm
    let re_percent = Regex::new(r"(\d+)\s*%").unwrap();
    s = re_percent
        .replace_all(&s, |caps: &regex::Captures| {
            let val: u64 = caps[1].parse().unwrap_or(0);
            format!("{} phần trăm", int_to_vietnamese_words(val))
        })
        .to_string();

    // 3. Chuẩn hóa các số nguyên thông thường (ví dụ: 100, 24, 2026,...)
    // Không match nếu nằm trong từ ngữ liền kề
    let re_numbers = Regex::new(r"\b\d+\b").unwrap();
    s = re_numbers
        .replace_all(&s, |caps: &regex::Captures| {
            if let Ok(val) = caps[0].parse::<u64>() {
                int_to_vietnamese_words(val)
            } else {
                caps[0].to_string()
            }
        })
        .to_string();

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numbers() {
        assert_eq!(int_to_vietnamese_words(0), "không");
        assert_eq!(int_to_vietnamese_words(5), "năm");
        assert_eq!(int_to_vietnamese_words(10), "mười");
        assert_eq!(int_to_vietnamese_words(15), "mười lăm");
        assert_eq!(int_to_vietnamese_words(21), "hai mươi mốt");
        assert_eq!(int_to_vietnamese_words(24), "hai mươi tư");
        assert_eq!(int_to_vietnamese_words(25), "hai mươi lăm");
        assert_eq!(int_to_vietnamese_words(100), "một trăm");
        assert_eq!(int_to_vietnamese_words(101), "một trăm linh một");
        assert_eq!(int_to_vietnamese_words(105), "một trăm linh năm");
        assert_eq!(int_to_vietnamese_words(115), "một trăm mười lăm");
        assert_eq!(int_to_vietnamese_words(125), "một trăm hai mươi lăm");
        assert_eq!(int_to_vietnamese_words(1000), "một nghìn");
        assert_eq!(int_to_vietnamese_words(2026), "hai nghìn không trăm hai mươi sáu");
    }

    #[test]
    fn test_sentence_norm() {
        let input = "Tôi đã vượt qua 100 tầng tháp trong năm 2026 với 99% công lực.";
        let out = normalize_vietnamese_text(input);
        assert!(out.contains("một trăm tầng"));
        assert!(out.contains("hai nghìn không trăm hai mươi sáu"));
        assert!(out.contains("chín mươi chín phần trăm"));
    }
}
