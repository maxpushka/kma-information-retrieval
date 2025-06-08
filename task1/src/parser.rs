use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

pub struct FB2Parser {
    word_regex: Regex,
}

impl FB2Parser {
    pub fn new() -> Self {
        FB2Parser {
            word_regex: Regex::new(r"\b[а-яёА-ЯЁa-zA-Z]{3,}\b").unwrap(),
        }
    }

    pub fn parse_file(&self, path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.trim_text(true);

        let mut words = Vec::new();
        let mut buf = Vec::new();
        let mut in_body = false;

        loop {
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"body" {
                        in_body = true;
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"body" {
                        in_body = false;
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_body {
                        let text = e.unescape()?;
                        for word_match in self.word_regex.find_iter(&text) {
                            let word = word_match.as_str().to_lowercase();
                            if word.len() >= 3 {
                                words.push(word);
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    eprintln!("Error parsing {}: {}", path.display(), e);
                    break;
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(words)
    }
}