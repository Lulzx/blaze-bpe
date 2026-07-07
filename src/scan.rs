//! Hand-rolled pre-tokenization scanners.
//!
//! These implement the exact piece-splitting semantics of tiktoken's regex
//! patterns (ordered alternation, possessive quantifiers, backtracking of
//! greedy quantifiers) as direct byte-level scanners, avoiding the cost of a
//! backtracking regex engine on the hot path.
//!
//! cl100k_base pattern:
//! `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`
//!
//! o200k_base pattern:
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|`
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|`
//! `\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

use unicode_general_category::{get_general_category, GeneralCategory as GC};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pattern {
    Cl100k,
    O200k,
}

// Character class bit flags.
const L: u8 = 1; // \p{L}
const N: u8 = 2; // \p{N}
const WS: u8 = 4; // \s (White_Space)
const NL: u8 = 8; // literal \r or \n
const UC: u8 = 16; // o200k "upper" class: Lu Lt Lm Lo M
const LC: u8 = 32; // o200k "lower" class: Ll Lm Lo M

const fn build_ascii_table() -> [u8; 128] {
    let mut t = [0u8; 128];
    let mut i = 0usize;
    while i < 128 {
        let c = i as u8;
        t[i] = if c >= b'A' && c <= b'Z' {
            L | UC
        } else if c >= b'a' && c <= b'z' {
            L | LC
        } else if c >= b'0' && c <= b'9' {
            N
        } else if c == b'\n' || c == b'\r' {
            WS | NL
        } else if c == b' ' || c == b'\t' || c == 0x0b || c == 0x0c {
            WS
        } else {
            0
        };
        i += 1;
    }
    t
}

static ASCII_CLASS: [u8; 128] = build_ascii_table();

#[inline]
fn class_of(c: char) -> u8 {
    let u = c as u32;
    if u < 128 {
        return ASCII_CLASS[u as usize];
    }
    match get_general_category(c) {
        GC::UppercaseLetter | GC::TitlecaseLetter => L | UC,
        GC::LowercaseLetter => L | LC,
        GC::ModifierLetter | GC::OtherLetter => L | UC | LC,
        // Marks are in both o200k word classes but are *not* \p{L}.
        GC::NonspacingMark | GC::SpacingMark | GC::EnclosingMark => UC | LC,
        GC::DecimalNumber | GC::LetterNumber | GC::OtherNumber => N,
        _ => {
            if c.is_whitespace() {
                WS
            } else {
                0
            }
        }
    }
}

/// Decode the char starting at byte `i` (must be a char boundary, i < s.len()).
#[inline]
fn char_at(s: &str, i: usize) -> (char, usize) {
    let b = s.as_bytes()[i];
    if b < 0x80 {
        (b as char, 1)
    } else {
        // SAFETY: caller guarantees `i` is a char boundary within `s`.
        let c = unsafe { s.get_unchecked(i..) }.chars().next().unwrap();
        (c, c.len_utf8())
    }
}

/// Match `(?i:s|t|m|d|ll|ve|re)` at byte offset `j` (just past an ASCII `'`).
/// This is the contraction-suffix set shared by cl100k and o200k.
#[inline]
fn contraction_suffix(b: &[u8], j: usize) -> Option<usize> {
    if j >= b.len() {
        return None;
    }
    // `| 0x20` lowercases ASCII letters; it cannot alias a UTF-8 lead or
    // continuation byte (>= 0x80) onto an ASCII letter.
    match b[j] | 0x20 {
        b's' | b'd' | b'm' | b't' => Some(j + 1),
        b'l' if j + 1 < b.len() && (b[j + 1] | 0x20) == b'l' => Some(j + 2),
        b'v' if j + 1 < b.len() && (b[j + 1] | 0x20) == b'e' => Some(j + 2),
        b'r' if j + 1 < b.len() && (b[j + 1] | 0x20) == b'e' => Some(j + 2),
        _ => None,
    }
}

/// Consume `\p{L}+` starting at byte `e`.
#[inline]
fn scan_letters(s: &str, mut e: usize) -> usize {
    let b = s.as_bytes();
    let n = b.len();
    loop {
        // ASCII letter fast loop: (c | 0x20) maps A-Z to a-z; anything >= 0x80
        // stays >= 0x80 and fails the range check.
        while e < n && (b[e] | 0x20).wrapping_sub(b'a') < 26 {
            e += 1;
        }
        if e >= n || b[e] < 0x80 {
            return e;
        }
        let (c, l) = char_at(s, e);
        if class_of(c) & L == 0 {
            return e;
        }
        e += l;
    }
}

/// Whitespace alternatives shared by both patterns, entered with a whitespace
/// char at `i`:
///   `\s*[\r\n]+` (or `[\r\n]`) → run up to and including its last \r/\n
///   `\s+(?!\S)`               → whole run when at end of text, else run minus
///                               its final char (which glues onto the next piece)
///   `\s+`                     → single whitespace char fallback
#[inline]
fn scan_whitespace(s: &str, i: usize, k0: u8, l0: usize) -> usize {
    let b = s.as_bytes();
    let n = s.len();
    let mut e = i + l0;
    let mut last_nl_end = if k0 & NL != 0 { e } else { 0 };
    let mut last_start = i;
    loop {
        // ASCII whitespace fast loop.
        while e < n && b[e] < 0x80 {
            let k = ASCII_CLASS[b[e] as usize];
            if k & WS == 0 {
                return finish_whitespace(i, e, n, last_nl_end, last_start);
            }
            last_start = e;
            e += 1;
            if k & NL != 0 {
                last_nl_end = e;
            }
        }
        if e >= n {
            return finish_whitespace(i, e, n, last_nl_end, last_start);
        }
        let (c, l) = char_at(s, e);
        if class_of(c) & WS == 0 {
            return finish_whitespace(i, e, n, last_nl_end, last_start);
        }
        last_start = e;
        e += l;
    }
}

#[inline]
fn finish_whitespace(i: usize, e: usize, n: usize, last_nl_end: usize, last_start: usize) -> usize {
    if last_nl_end > 0 {
        return last_nl_end;
    }
    if e >= n {
        return e;
    }
    if last_start > i {
        return last_start;
    }
    e
}

/// Punct alternative ` ?[^\s\p{L}\p{N}]+[\r\n]*` (o200k adds `/` to the tail
/// class, which is unreachable there — `/` is always consumed by the `+` run).
/// Returns None if the alternative does not match at `i`.
#[inline]
fn scan_punct(s: &str, i: usize, c0: char, k0: u8, l0: usize) -> Option<usize> {
    let n = s.len();
    let (pstart, pk, pl) = if c0 == ' ' {
        if i + 1 >= n {
            return None;
        }
        let (c1, l1) = char_at(s, i + 1);
        (i + 1, class_of(c1), l1)
    } else {
        (i, k0, l0)
    };
    if pk & (L | N | WS) != 0 {
        return None;
    }
    let b = s.as_bytes();
    let mut e = pstart + pl;
    loop {
        while e < n && b[e] < 0x80 && ASCII_CLASS[b[e] as usize] & (L | N | WS) == 0 {
            e += 1;
        }
        if e >= n || b[e] < 0x80 {
            break;
        }
        let (c, l) = char_at(s, e);
        if class_of(c) & (L | N | WS) != 0 {
            break;
        }
        e += l;
    }
    while e < n && (b[e] == b'\n' || b[e] == b'\r') {
        e += 1;
    }
    Some(e)
}

/// `\p{N}{1,3}` with a digit already matched at `i`.
#[inline]
fn scan_digits(s: &str, i: usize, l0: usize) -> usize {
    let n = s.len();
    let mut e = i + l0;
    let mut cnt = 1;
    while cnt < 3 && e < n {
        let (c, l) = char_at(s, e);
        if class_of(c) & N == 0 {
            break;
        }
        e += l;
        cnt += 1;
    }
    e
}

/// Find the end of the piece starting at byte `i` of `s` (cl100k pattern).
pub fn next_piece_cl100k(s: &str, i: usize) -> usize {
    let n = s.len();
    let (c0, l0) = char_at(s, i);
    let k0 = class_of(c0);

    // Alt 1: '(?i:[sdmt]|ll|ve|re)
    if c0 == '\'' {
        if let Some(e) = contraction_suffix(s.as_bytes(), i + 1) {
            return e;
        }
    }

    // Alt 2: [^\r\n\p{L}\p{N}]?+\p{L}+
    if k0 & L != 0 {
        return scan_letters(s, i + l0);
    }
    if k0 & (NL | N) == 0 {
        let j = i + l0;
        if j < n {
            let (c1, l1) = char_at(s, j);
            if class_of(c1) & L != 0 {
                return scan_letters(s, j + l1);
            }
        }
    }

    // Alt 3: \p{N}{1,3}
    if k0 & N != 0 {
        return scan_digits(s, i, l0);
    }

    // Alt 4: ` ?[^\s\p{L}\p{N}]++[\r\n]*`
    if let Some(e) = scan_punct(s, i, c0, k0, l0) {
        return e;
    }

    // Alts 5-7: whitespace. Reaching here implies c0 is whitespace.
    debug_assert!(k0 & WS != 0);
    scan_whitespace(s, i, k0, l0)
}

/// o200k word body `A* B+` (alt 1) starting at `p`, where A = [Lu Lt Lm Lo M]
/// and B = [Ll Lm Lo M]. Implements greedy `A*` with backtracking: after the
/// maximal A-run, `B+` is tried at the run end, then at successively earlier
/// run chars until one is in B.
fn o200k_alt1_body(s: &str, p: usize) -> Option<usize> {
    let n = s.len();
    // Record the maximal A-run and the start offset of each of its chars.
    let mut e = p;
    let mut starts: smallvec::SmallVec<[u32; 32]> = smallvec::SmallVec::new();
    while e < n {
        let (c, l) = char_at(s, e);
        if class_of(c) & UC == 0 {
            break;
        }
        starts.push(e as u32);
        e += l;
    }
    // Try B+ at the end of the A-run, then backtrack into it.
    let mut try_at = e;
    loop {
        // B+ requires at least one B char at try_at.
        if try_at < n {
            let (c, l) = char_at(s, try_at);
            if class_of(c) & LC != 0 {
                // Greedy B+ from here.
                let mut be = try_at + l;
                while be < n {
                    let (c2, l2) = char_at(s, be);
                    if class_of(c2) & LC == 0 {
                        break;
                    }
                    be += l2;
                }
                return Some(be);
            }
        }
        match starts.pop() {
            Some(st) => try_at = st as usize,
            None => return None,
        }
    }
}

/// o200k word body `A+ B*` (alt 2) starting at `p`.
fn o200k_alt2_body(s: &str, p: usize) -> Option<usize> {
    let n = s.len();
    let mut e = p;
    while e < n {
        let (c, l) = char_at(s, e);
        if class_of(c) & UC == 0 {
            break;
        }
        e += l;
    }
    if e == p {
        return None;
    }
    while e < n {
        let (c, l) = char_at(s, e);
        if class_of(c) & LC == 0 {
            break;
        }
        e += l;
    }
    Some(e)
}

/// Apply optional prefix `[^\r\n\p{L}\p{N}]?` (greedy, non-possessive: retried
/// without the prefix on failure) plus a word body, plus the optional
/// contraction suffix.
#[inline]
fn o200k_word(
    s: &str,
    i: usize,
    _c0: char,
    k0: u8,
    l0: usize,
    body: fn(&str, usize) -> Option<usize>,
) -> Option<usize> {
    let end = if k0 & (L | N | NL) == 0 {
        // Prefix char matches; try with it first, then without.
        body(s, i + l0).or_else(|| body(s, i))?
    } else {
        body(s, i)?
    };
    let b = s.as_bytes();
    if end < s.len() && b[end] == b'\'' {
        if let Some(e2) = contraction_suffix(b, end + 1) {
            return Some(e2);
        }
    }
    Some(end)
}

/// Find the end of the piece starting at byte `i` of `s` (o200k pattern).
pub fn next_piece_o200k(s: &str, i: usize) -> usize {
    let (c0, l0) = char_at(s, i);
    let k0 = class_of(c0);

    // Fast path: lowercase-ish start (the overwhelmingly common case) can
    // only match alt 1 with empty prefix and empty A*.
    if k0 & LC != 0 && k0 & UC == 0 {
        let b = s.as_bytes();
        let n = s.len();
        let mut e = i + l0;
        loop {
            // ASCII lowercase fast loop.
            while e < n && b[e].wrapping_sub(b'a') < 26 {
                e += 1;
            }
            if e >= n || b[e] < 0x80 {
                break;
            }
            let (c, l) = char_at(s, e);
            if class_of(c) & LC == 0 {
                break;
            }
            e += l;
        }
        if e < n && b[e] == b'\'' {
            if let Some(e2) = contraction_suffix(b, e + 1) {
                return e2;
            }
        }
        return e;
    }

    // Alt 1 then Alt 2 (word forms).
    if let Some(e) = o200k_word(s, i, c0, k0, l0, o200k_alt1_body) {
        return e;
    }
    if let Some(e) = o200k_word(s, i, c0, k0, l0, o200k_alt2_body) {
        return e;
    }

    // Alt 3: \p{N}{1,3}
    if k0 & N != 0 {
        return scan_digits(s, i, l0);
    }

    // Alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
    if let Some(e) = scan_punct(s, i, c0, k0, l0) {
        return e;
    }

    // Alts 5-7: whitespace.
    debug_assert!(k0 & WS != 0);
    scan_whitespace(s, i, k0, l0)
}

/// Split `s` into pretokenization pieces, invoking `f(start, end)` per piece.
#[inline]
pub fn for_each_piece(s: &str, pattern: Pattern, mut f: impl FnMut(usize, usize)) {
    let mut i = 0;
    let n = s.len();
    match pattern {
        Pattern::Cl100k => {
            while i < n {
                let e = next_piece_cl100k(s, i);
                debug_assert!(e > i);
                f(i, e);
                i = e;
            }
        }
        Pattern::O200k => {
            while i < n {
                let e = next_piece_o200k(s, i);
                debug_assert!(e > i);
                f(i, e);
                i = e;
            }
        }
    }
}
