//! Subtitle generation (.srt) with precision timestamps for sentences.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SubtitleItem {
    pub id: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub text: String,
}

/// Chuyển đổi số giây thành định dạng SRT: HH:MM:SS,mmm
pub fn format_srt_time(seconds: f64) -> String {
    let millis = ((seconds.fract() * 1000.0).round() as u64) % 1000;
    let total_secs = seconds.floor() as u64;
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = total_secs / 3600;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
}

/// Ghi danh sách phụ đề ra file chuẩn .srt
pub fn write_srt_file<P: AsRef<Path>>(items: &[SubtitleItem], path: P) -> io::Result<()> {
    let mut file = File::create(path)?;
    for item in items {
        let start_str = format_srt_time(item.start_sec);
        let end_str = format_srt_time(item.end_sec);
        writeln!(file, "{}", item.id)?;
        writeln!(file, "{} --> {}", start_str, end_str)?;
        writeln!(file, "{}", item.text)?;
        writeln!(file)?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.234), "00:00:01,234");
        assert_eq!(format_srt_time(65.5), "00:01:05,500");
        assert_eq!(format_srt_time(3661.050), "01:01:01,050");
    }
}
