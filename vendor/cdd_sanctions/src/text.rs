//! Name normalisation, tokenisation and phonetic skeletons.

use crate::model::SubjectKind;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

fn cyrillic(c: char) -> Option<&'static str> {
    Some(match c {
        'а' => "a", 'б' => "b", 'в' => "v", 'г' => "g", 'д' => "d", 'е' => "e", 'ё' => "e",
        'ж' => "zh", 'з' => "z", 'и' => "i", 'й' => "y", 'к' => "k", 'л' => "l", 'м' => "m",
        'н' => "n", 'о' => "o", 'п' => "p", 'р' => "r", 'с' => "s", 'т' => "t", 'у' => "u",
        'ф' => "f", 'х' => "kh", 'ц' => "ts", 'ч' => "ch", 'ш' => "sh", 'щ' => "shch",
        'ъ' => "", 'ы' => "y", 'ь' => "", 'э' => "e", 'ю' => "yu", 'я' => "ya",
        'і' => "i", 'ї' => "yi", 'є' => "ye", 'ґ' => "g",
        _ => return None,
    })
}

/// Lower-case, strip diacritics, transliterate Cyrillic, drop apostrophes,
/// turn every other non-alphanumeric into a space and collapse whitespace.
pub fn fold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.nfkd() {
        if is_combining_mark(c) {
            continue;
        }
        for lc in c.to_lowercase() {
            if let Some(t) = cyrillic(lc) {
                out.push_str(t);
                continue;
            }
            match lc {
                '\'' | '\u{2019}' | '\u{2018}' | '`' => {}
                'ß' => out.push_str("ss"),
                'æ' => out.push_str("ae"),
                'œ' => out.push_str("oe"),
                'ø' => out.push('o'),
                'đ' | 'ð' => out.push('d'),
                'þ' => out.push_str("th"),
                'ł' => out.push('l'),
                'ı' => out.push('i'),
                c if c.is_alphanumeric() => out.push(c),
                _ => out.push(' '),
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

const HONORIFICS: &[&str] = &["mr", "mrs", "ms", "miss", "dr", "prof", "sir", "hon", "eng"];
const LEGAL_FORMS: &[&str] = &[
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "sa", "sarl", "gmbh", "ag", "bv", "nv", "plc", "llp", "lp", "pte", "pvt", "jsc", "ooo",
    "zao", "oao", "pjsc", "ojsc", "srl", "spa", "sas", "kg", "ltda", "fze", "fzc", "fzco",
    "dmcc", "asa", "sdn", "bhd",
];

/// Tokenise a name, removing honorifics (people) and legal-form suffixes
/// (entities). Falls back to the unfiltered tokens if filtering empties it.
pub fn tokens(name: &str, kind: SubjectKind) -> Vec<String> {
    let folded = fold(name);
    let all: Vec<String> = folded.split_whitespace().map(str::to_string).collect();
    let strip_h = matches!(kind, SubjectKind::Individual | SubjectKind::Unknown);
    let strip_l = matches!(kind, SubjectKind::Entity | SubjectKind::Unknown);
    let filtered: Vec<String> = all
        .iter()
        .filter(|t| !(strip_h && HONORIFICS.contains(&t.as_str())))
        .filter(|t| !(strip_l && LEGAL_FORMS.contains(&t.as_str())))
        .cloned()
        .collect();
    if filtered.is_empty() {
        all
    } else {
        filtered
    }
}

/// Consonant skeleton: collapses common transliteration variants
/// (Mohammed / Muhammad / Mohamed all become "mmd").
pub fn skeleton(tok: &str) -> String {
    let mut s = tok.to_string();
    for (a, b) in [
        ("ph", "f"), ("ck", "k"), ("kh", "k"), ("gh", "g"), ("dh", "d"), ("th", "t"),
        ("sh", "s"), ("ch", "s"), ("qu", "k"), ("x", "ks"), ("q", "k"), ("z", "s"),
        ("w", "v"), ("c", "k"),
    ] {
        s = s.replace(a, b);
    }
    // collapse doubled letters, then drop non-initial vowels and h
    let mut dedup = String::with_capacity(s.len());
    let mut prev = None;
    for c in s.chars() {
        if Some(c) != prev {
            dedup.push(c);
        }
        prev = Some(c);
    }
    let mut out = String::with_capacity(dedup.len());
    for (i, c) in dedup.chars().enumerate() {
        if i > 0 && "aeiouyh".contains(c) {
            continue;
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_basics() {
        assert_eq!(fold("  José  O'Brien-Ålesund "), "jose obrien alesund");
        assert_eq!(fold("Иван Петров"), "ivan petrov");
        assert_eq!(fold("Straße"), "strasse");
    }

    #[test]
    fn strips_legal_forms_and_honorifics() {
        assert_eq!(tokens("Northern Star Trading LLC", SubjectKind::Entity), vec!["northern", "star", "trading"]);
        assert_eq!(tokens("Dr. Ivan Petrov", SubjectKind::Individual), vec!["ivan", "petrov"]);
        // never empties a name entirely
        assert_eq!(tokens("Co Ltd", SubjectKind::Entity), vec!["co", "ltd"]);
    }

    #[test]
    fn skeleton_collapses_variants() {
        let a = skeleton("mohammed");
        assert_eq!(a, skeleton("muhammad"));
        assert_eq!(a, skeleton("mohamed"));
        assert_eq!(skeleton("ahmed"), skeleton("ahmad"));
        assert_ne!(skeleton("petrov"), skeleton("ivanov"));
    }
}
