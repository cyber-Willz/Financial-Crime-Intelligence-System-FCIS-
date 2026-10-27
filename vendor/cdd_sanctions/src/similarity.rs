//! String and name similarity.

use crate::text::skeleton;

pub fn jaro(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dist = (a.len().max(b.len()) / 2).saturating_sub(1);
    let mut am = vec![false; a.len()];
    let mut bm = vec![false; b.len()];
    let mut matches = 0usize;
    for i in 0..a.len() {
        let lo = i.saturating_sub(dist);
        let hi = (i + dist + 1).min(b.len());
        for j in lo..hi {
            if !bm[j] && a[i] == b[j] {
                am[i] = true;
                bm[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut t = 0usize;
    let mut k = 0usize;
    for i in 0..a.len() {
        if am[i] {
            while !bm[k] {
                k += 1;
            }
            if a[i] != b[k] {
                t += 1;
            }
            k += 1;
        }
    }
    let m = matches as f64;
    (m / a.len() as f64 + m / b.len() as f64 + (m - t as f64 / 2.0) / m) / 3.0
}

pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let j = jaro(&ac, &bc);
    if j <= 0.7 {
        return j;
    }
    let l = ac.iter().zip(bc.iter()).take(4).take_while(|(x, y)| x == y).count();
    j + l as f64 * 0.1 * (1.0 - j)
}

/// Similarity of two single name tokens in [0,1].
pub fn token_sim(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let (la, lb) = (a.chars().count(), b.chars().count());
    if la == 1 || lb == 1 {
        // initial vs full token
        let (short, long) = if la == 1 { (a, b) } else { (b, a) };
        return if long.starts_with(short) { 0.7 } else { 0.0 };
    }
    let jw = jaro_winkler(a, b);
    let (sa, sb) = (skeleton(a), skeleton(b));
    // Phonetic credit needs real words on both sides: a 2-letter token ("RF", "AG", "VEB.RF" split on the dot) is
    // usually an abbreviation, and its consonant skeleton would otherwise equal that of any word with the same
    // consonants ("reef" -> "rf"), manufacturing a "strong" match that defeats the single-shared-word cap.
    let ph = if sa == sb && sa.chars().count() >= 2 && la.min(lb) >= 3 { 0.92 } else { 0.0 };
    jw.max(ph)
}

/// A name prepared once for repeated comparison.
#[derive(Clone, Debug, Default)]
pub struct PreparedName {
    pub tokens: Vec<String>,
    pub sorted_joined: String,
}

impl PreparedName {
    pub fn new(tokens: Vec<String>) -> Self {
        let mut s = tokens.clone();
        s.sort();
        Self { sorted_joined: s.concat(), tokens }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NameScore {
    pub score: f64,
    pub strong_matches: usize,
}

const STRONG_TOKEN: f64 = 0.88;

/// Order-insensitive, length-weighted token alignment between two names.
pub fn name_score(q: &PreparedName, c: &PreparedName) -> NameScore {
    let (nq, nc) = (q.tokens.len(), c.tokens.len());
    if nq == 0 || nc == 0 {
        return NameScore { score: 0.0, strong_matches: 0 };
    }
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (i, qt) in q.tokens.iter().enumerate() {
        for (j, ct) in c.tokens.iter().enumerate() {
            let s = token_sim(qt, ct);
            if s >= 0.6 {
                pairs.push((s, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let (mut qu, mut cu) = (vec![false; nq], vec![false; nc]);
    let w = |t: &String| t.chars().count().max(1) as f64;
    let (mut sq, mut sc, mut strong) = (0.0, 0.0, 0usize);
    for (s, i, j) in pairs {
        if qu[i] || cu[j] {
            continue;
        }
        qu[i] = true;
        cu[j] = true;
        sq += s * w(&q.tokens[i]);
        sc += s * w(&c.tokens[j]);
        if s >= STRONG_TOKEN {
            strong += 1;
        }
    }
    let tq: f64 = q.tokens.iter().map(w).sum();
    let tc: f64 = c.tokens.iter().map(w).sum();
    let mut score = 0.75 * (sq / tq) + 0.25 * (sc / tc);

    // spacing/hyphenation differences ("al qaida" vs "alqaida")
    let (lq, lc) = (q.sorted_joined.chars().count(), c.sorted_joined.chars().count());
    let mut joined_used = false;
    if lq.abs_diff(lc) <= 2 {
        let j = jaro_winkler(&q.sorted_joined, &c.sorted_joined);
        if j >= 0.93 && j * 0.97 > score {
            score = j * 0.97;
            joined_used = true;
        }
    }

    // a single shared word (typically a common surname) is not enough
    let exact_single = nq == 1 && nc == 1 && q.tokens[0] == c.tokens[0];
    if !joined_used && !exact_single && (nq.min(nc) == 1 && nq.max(nc) > 1 || (nq >= 2 && nc >= 2 && strong < 2)) {
        score = score.min(0.80);
    }
    NameScore { score: score.clamp(0.0, 1.0), strong_matches: strong }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SubjectKind::Individual;
    use crate::text::tokens;

    fn p(s: &str) -> PreparedName {
        PreparedName::new(tokens(s, Individual))
    }

    #[test]
    fn short_abbreviations_do_not_phonetically_match_words() {
        // regression from real OFAC data: "Reef Developments Ltd" scored 0.88 against the alias
        // "STATE DEVELOPMENT CORPORATION VEB.RF" because skeleton("reef") == skeleton("rf")
        assert!(token_sim("reef", "rf") < 0.88, "{}", token_sim("reef", "rf"));
        let q = PreparedName::new(tokens("Reef Developments Ltd", crate::model::SubjectKind::Entity));
        let c = PreparedName::new(tokens("STATE DEVELOPMENT CORPORATION VEB.RF", crate::model::SubjectKind::Entity));
        let s = name_score(&q, &c);
        assert!(s.score <= 0.80 && s.strong_matches < 2, "{s:?}");
        // genuine transliteration variants keep their phonetic credit
        assert!(token_sim("kim", "kam") >= 0.92);
        assert!(token_sim("mohammed", "muhammad") >= 0.92);
        // and the same shared word plus a genuinely matching second word is still a strong match
        let c2 = PreparedName::new(tokens("STATE DEVELOPMENT CORPORATION", crate::model::SubjectKind::Entity));
        let q2 = PreparedName::new(tokens("State Development Corporation", crate::model::SubjectKind::Entity));
        assert!(name_score(&q2, &c2).score > 0.95);
    }

    #[test]
    fn jw_reference_values() {
        assert!((jaro_winkler("martha", "marhta") - 0.9611).abs() < 1e-3);
        assert!((jaro_winkler("dixon", "dicksonx") - 0.8133).abs() < 1e-3);
        assert_eq!(jaro_winkler("abc", "abc"), 1.0);
    }

    #[test]
    fn order_and_variant_insensitive() {
        assert!(name_score(&p("Vladimir Petrov"), &p("PETROV, Vladimir")).score > 0.99);
        assert!(name_score(&p("Mohammed Al Hassan"), &p("Muhammad Alhassan")).score > 0.9);
        assert!(name_score(&p("Petrova Ivanovich"), &p("Petrov Ivanovic")).score > 0.9);
    }

    #[test]
    fn missing_middle_name_still_scores_for_review() {
        let s = name_score(&p("Ivan Petrov"), &p("PETROV, Ivan Sergeyevich")).score;
        assert!(s > 0.85 && s < 1.0, "{s}");
    }

    #[test]
    fn shared_surname_only_is_capped() {
        assert!(name_score(&p("Adam Smith"), &p("John Smith")).score <= 0.80);
        assert!(name_score(&p("Smith"), &p("John Smith")).score <= 0.80);
    }

    #[test]
    fn unrelated_names_are_low() {
        assert!(name_score(&p("Maria Gonzalez"), &p("Chen Wei")).score < 0.5);
    }

    #[test]
    fn typo_tolerance() {
        assert!(name_score(&p("Ivan Petrof"), &p("Ivan Petrov")).score > 0.9);
    }
}
