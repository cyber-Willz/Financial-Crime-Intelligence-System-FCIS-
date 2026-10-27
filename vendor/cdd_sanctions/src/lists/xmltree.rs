//! Minimal XML tree on top of quick-xml (lists are a few MB; a DOM is fine).

use crate::error::{Error, Result};
use quick_xml::events::Event;
use quick_xml::Reader;

#[derive(Debug, Default, Clone)]
pub struct Node {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<Node>,
}

impl Node {
    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }
    pub fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    /// Trimmed text of the first child with this name, if non-empty.
    pub fn text_of(&self, name: &str) -> Option<String> {
        let t = self.child(name)?.text.trim().to_string();
        (!t.is_empty()).then_some(t)
    }
    /// Trimmed texts of all grandchildren `name` under every child `container`.
    pub fn values_under(&self, container: &str, name: &str) -> Vec<String> {
        self.all(container)
            .flat_map(|c| c.all(name))
            .map(|n| n.text.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect()
    }
}

fn local(name: &[u8]) -> String {
    let s = String::from_utf8_lossy(name);
    s.rsplit(':').next().unwrap_or(&s).to_string()
}

fn open(e: &quick_xml::events::BytesStart) -> Result<Node> {
    let mut n = Node { name: local(e.name().as_ref()), ..Default::default() };
    for a in e.attributes() {
        let a = a.map_err(|x| Error::Xml(x.to_string()))?;
        let v = a.unescape_value().map_err(|x| Error::Xml(x.to_string()))?;
        n.attrs.push((local(a.key.as_ref()), v.into_owned()));
    }
    Ok(n)
}

pub fn parse(xml: &str) -> Result<Node> {
    let mut r = Reader::from_str(xml);
    r.trim_text(true);
    let mut stack: Vec<Node> = Vec::new();
    let mut root: Option<Node> = None;
    loop {
        match r.read_event().map_err(|e| Error::Xml(e.to_string()))? {
            Event::Start(e) => stack.push(open(&e)?),
            Event::Empty(e) => {
                let n = open(&e)?;
                match stack.last_mut() {
                    Some(p) => p.children.push(n),
                    None => root = Some(n),
                }
            }
            Event::Text(t) => {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&t.unescape().map_err(|e| Error::Xml(e.to_string()))?);
                }
            }
            Event::CData(c) => {
                if let Some(top) = stack.last_mut() {
                    top.text.push_str(&String::from_utf8_lossy(&c.into_inner()));
                }
            }
            Event::End(_) => {
                let n = stack.pop().ok_or_else(|| Error::Xml("unbalanced end tag".into()))?;
                match stack.last_mut() {
                    Some(p) => p.children.push(n),
                    None => root = Some(n),
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err(Error::Xml("unexpected end of document".into()));
    }
    root.ok_or_else(|| Error::Xml("empty document".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builds_tree() {
        let n = parse(r#"<a x="1"><b>hi</b><b>yo</b><c/><ns:d k="v"/></a>"#).unwrap();
        assert_eq!(n.attr("x"), Some("1"));
        assert_eq!(n.all("b").count(), 2);
        assert_eq!(n.text_of("b").as_deref(), Some("hi"));
        assert_eq!(n.child("d").unwrap().attr("k"), Some("v"));
        assert!(parse("<a><b></a>").is_err());
    }
}
