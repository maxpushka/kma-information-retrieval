use bit_vec::BitVec;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::dictionary::Dictionary;
use crate::query::{tokenize, QueryParser};

#[derive(Debug, Serialize, Deserialize)]
pub struct IncidenceMatrix {
    pub terms: Vec<String>,
    pub documents: Vec<String>,
    pub matrix: Vec<BitVec>,
}

impl IncidenceMatrix {
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        let mut terms: Vec<String> = dictionary.terms.keys().cloned().collect();
        terms.sort();
        
        let mut documents: HashSet<String> = HashSet::new();
        for term_entry in dictionary.terms.values() {
            for doc in &term_entry.documents {
                documents.insert(doc.clone());
            }
        }
        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();
        
        let mut matrix = Vec::new();
        for term in &terms {
            let mut row = BitVec::from_elem(documents.len(), false);
            if let Some(term_entry) = dictionary.terms.get(term) {
                for doc in &term_entry.documents {
                    if let Some(doc_idx) = documents.iter().position(|d| d == doc) {
                        row.set(doc_idx, true);
                    }
                }
            }
            matrix.push(row);
        }
        
        IncidenceMatrix {
            terms,
            documents,
            matrix,
        }
    }
    
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.terms.iter().map(|t| t.len()).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>() +
        self.matrix.iter().map(|row| row.capacity() / 8).sum::<usize>()
    }

    fn parse_or_expr(&self, tokens: &[String], pos: &mut usize) -> Result<BitVec, String> {
        let mut result = self.parse_and_expr(tokens, pos)?;
        
        while *pos < tokens.len() && tokens[*pos] == "or" {
            *pos += 1;
            let right = self.parse_and_expr(tokens, pos)?;
            result.or(&right);
        }
        
        Ok(result)
    }
    
    fn parse_and_expr(&self, tokens: &[String], pos: &mut usize) -> Result<BitVec, String> {
        let mut result = self.parse_not_expr(tokens, pos)?;
        
        while *pos < tokens.len() && tokens[*pos] == "and" {
            *pos += 1;
            let right = self.parse_not_expr(tokens, pos)?;
            result.and(&right);
        }
        
        Ok(result)
    }
    
    fn parse_not_expr(&self, tokens: &[String], pos: &mut usize) -> Result<BitVec, String> {
        if *pos < tokens.len() && tokens[*pos] == "not" {
            *pos += 1;
            let mut result = self.parse_primary(tokens, pos)?;
            result.negate();
            Ok(result)
        } else {
            self.parse_primary(tokens, pos)
        }
    }
    
    fn parse_primary(&self, tokens: &[String], pos: &mut usize) -> Result<BitVec, String> {
        if *pos >= tokens.len() {
            return Err("Unexpected end of query".to_string());
        }
        
        if tokens[*pos] == "(" {
            *pos += 1;
            let result = self.parse_or_expr(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err("Missing closing parenthesis".to_string());
            }
            *pos += 1;
            Ok(result)
        } else {
            let term = &tokens[*pos];
            *pos += 1;
            self.search_term(term)
        }
    }
    
    pub fn get_matching_documents(&self, result: &BitVec) -> Vec<&String> {
        result.iter()
            .enumerate()
            .filter_map(|(idx, bit)| if bit { Some(&self.documents[idx]) } else { None })
            .collect()
    }
}

impl QueryParser for IncidenceMatrix {
    type Result = BitVec;
    type Error = String;

    fn search(&self, query: &str) -> Result<Self::Result, Self::Error> {
        let tokens = tokenize(&query.to_lowercase())?;
        let mut pos = 0;
        self.parse_or_expr(&tokens, &mut pos)
    }

    fn search_term(&self, term: &str) -> Result<Self::Result, Self::Error> {
        if let Some(term_idx) = self.terms.iter().position(|t| t == term) {
            Ok(self.matrix[term_idx].clone())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}