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

/// Advance `e` past ASCII letters (A-Z, a-z), 16 bytes at a time on aarch64.
/// `OR_CASE` folds case before the range test; with it false only a-z match.
#[inline]
fn ascii_run<const OR_CASE: bool>(b: &[u8], mut e: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while e + 16 <= b.len() {
            let mut v = vld1q_u8(b.as_ptr().add(e));
            if OR_CASE {
                v = vorrq_u8(v, vdupq_n_u8(0x20));
            }
            let is_letter = vcltq_u8(vsubq_u8(v, vdupq_n_u8(b'a')), vdupq_n_u8(26));
            // Narrowing-shift movemask: 4 mask bits per byte lane.
            let m = vshrn_n_u16(vreinterpretq_u16_u8(is_letter), 4);
            let mask = vget_lane_u64(vreinterpret_u64_u8(m), 0);
            if mask == u64::MAX {
                e += 16;
                continue;
            }
            return e + ((!mask).trailing_zeros() as usize >> 2);
        }
    }
    while e < b.len() {
        let c = if OR_CASE { b[e] | 0x20 } else { b[e] };
        if c.wrapping_sub(b'a') >= 26 {
            break;
        }
        e += 1;
    }
    e
}

/// Consume `\p{L}+` starting at byte `e`.
#[inline]
fn scan_letters(s: &str, mut e: usize) -> usize {
    let b = s.as_bytes();
    let n = b.len();
    loop {
        e = ascii_run::<true>(b, e);
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

/// Punct alternative ` ?[^\s\p{L}\p{N}]+[\r\n]*`, or with `SLASH` the o200k
/// form ` ?[^\s\p{L}\p{N}]+[\r\n/]*`. A `/` right after the `+` run is
/// consumed by the run itself, but one after a `\r`/`\n` is only reachable
/// through the tail class (`"//\n//"` is a single o200k piece).
/// Returns None if the alternative does not match at `i`.
#[inline]
fn scan_punct<const SLASH: bool>(s: &str, i: usize, c0: char, k0: u8, l0: usize) -> Option<usize> {
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
    while e < n && (b[e] == b'\n' || b[e] == b'\r' || (SLASH && b[e] == b'/')) {
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
    let b = s.as_bytes();
    // Fast paths for the two dominant piece shapes; exact per the pattern:
    // an ASCII letter can only start alt 2 (alt 1 needs `'`), and a space
    // followed by an ASCII letter can only match alt 2's optional prefix.
    let c = b[i];
    if (c | 0x20).wrapping_sub(b'a') < 26 {
        return scan_letters(s, i + 1);
    }
    if c == b' ' && i + 1 < n && (b[i + 1] | 0x20).wrapping_sub(b'a') < 26 {
        return scan_letters(s, i + 2);
    }
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
    if let Some(e) = scan_punct::<false>(s, i, c0, k0, l0) {
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

/// Greedy `B+` continuation from `e` plus optional contraction suffix
/// (o200k alt 1 with empty `A*`).
#[inline]
fn o200k_lower_run(s: &str, mut e: usize) -> usize {
    let b = s.as_bytes();
    let n = s.len();
    loop {
        e = ascii_run::<false>(b, e);
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
    e
}

/// Find the end of the piece starting at byte `i` of `s` (o200k pattern).
pub fn next_piece_o200k(s: &str, i: usize) -> usize {
    // Fast path: space followed by an ASCII lowercase letter can only match
    // alt 1 with the space as prefix and empty A*.
    {
        let b = s.as_bytes();
        if b[i] == b' ' && i + 1 < s.len() && b[i + 1].wrapping_sub(b'a') < 26 {
            return o200k_lower_run(s, i + 2);
        }
    }
    let (c0, l0) = char_at(s, i);
    let k0 = class_of(c0);

    // Fast path: lowercase-ish start (the overwhelmingly common case) can
    // only match alt 1 with empty prefix and empty A*.
    if k0 & LC != 0 && k0 & UC == 0 {
        return o200k_lower_run(s, i + l0);
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
    if let Some(e) = scan_punct::<true>(s, i, c0, k0, l0) {
        return e;
    }

    // Alts 5-7: whitespace.
    debug_assert!(k0 & WS != 0);
    scan_whitespace(s, i, k0, l0)
}

/// Byte-class bitmasks for a 64-byte window, one bit per byte. Piece-boundary
/// decisions become shift + count-trailing-zeros instead of per-byte branchy
/// loads, and one classification is shared by every piece in the window.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Default)]
struct ClassWin {
    /// ASCII letter (case-folded a-z)
    l: u64,
    /// ASCII digit
    d: u64,
    /// literal \r or \n
    nl: u64,
    /// ASCII whitespace: \t \n \v \f \r space
    ws: u64,
    /// non-ASCII (byte >= 0x80)
    na: u64,
    /// literal space
    sp: u64,
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn classify64(p: *const u8) -> ClassWin {
    use std::arch::aarch64::*;
    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    /// 1-bit-per-byte movemask over four 16-byte compare results.
    #[inline]
    unsafe fn mm64(
        m: [uint8x16_t; 4],
        bits: uint8x16_t,
    ) -> u64 {
        let t0 = vpaddq_u8(vandq_u8(m[0], bits), vandq_u8(m[1], bits));
        let t1 = vpaddq_u8(vandq_u8(m[2], bits), vandq_u8(m[3], bits));
        let t2 = vpaddq_u8(t0, t1);
        let t3 = vpaddq_u8(t2, t2);
        vgetq_lane_u64(vreinterpretq_u64_u8(t3), 0)
    }
    let bits = vld1q_u8(BITS.as_ptr());
    let v = [
        vld1q_u8(p),
        vld1q_u8(p.add(16)),
        vld1q_u8(p.add(32)),
        vld1q_u8(p.add(48)),
    ];
    let mut is_l = [vdupq_n_u8(0); 4];
    let mut is_d = [vdupq_n_u8(0); 4];
    let mut is_nl = [vdupq_n_u8(0); 4];
    let mut is_ws = [vdupq_n_u8(0); 4];
    let mut is_na = [vdupq_n_u8(0); 4];
    let mut is_sp = [vdupq_n_u8(0); 4];
    for k in 0..4 {
        let folded = vorrq_u8(v[k], vdupq_n_u8(0x20));
        is_l[k] = vcltq_u8(vsubq_u8(folded, vdupq_n_u8(b'a')), vdupq_n_u8(26));
        is_d[k] = vcltq_u8(vsubq_u8(v[k], vdupq_n_u8(b'0')), vdupq_n_u8(10));
        is_na[k] = vcgeq_u8(v[k], vdupq_n_u8(0x80));
        is_nl[k] = vorrq_u8(
            vceqq_u8(v[k], vdupq_n_u8(b'\n')),
            vceqq_u8(v[k], vdupq_n_u8(b'\r')),
        );
        is_sp[k] = vceqq_u8(v[k], vdupq_n_u8(b' '));
        is_ws[k] = vorrq_u8(
            is_sp[k],
            vcltq_u8(vsubq_u8(v[k], vdupq_n_u8(0x09)), vdupq_n_u8(5)),
        );
    }
    ClassWin {
        l: mm64(is_l, bits),
        d: mm64(is_d, bits),
        nl: mm64(is_nl, bits),
        ws: mm64(is_ws, bits),
        na: mm64(is_na, bits),
        sp: mm64(is_sp, bits),
    }
}

/// Mask-driven cl100k piece loop.
///
/// Runs of `letters (space letters)*` — the dominant shape in real text — are
/// emitted by a "pure lane": within the maximal region of the window that
/// contains only letters and word-leading spaces, every piece boundary is a
/// set bit in one precomputed start mask, so each piece costs a
/// count-trailing-zeros and a clear-lowest-bit, with zero dispatch branches.
/// Everything else dispatches off the class bitmasks (mirroring the
/// alternation order of the pattern); non-ASCII, contractions, and
/// whitespace runs fall back to the scalar `next_piece_cl100k`, which is
/// exact.
#[cfg(target_arch = "aarch64")]
fn for_each_piece_cl100k_masked(s: &str, mut f: impl FnMut(usize, usize)) {
    let b = s.as_bytes();
    let n = s.len();
    let mut i = 0usize;
    // Window covers [wbase, wbase + 64); refilled so every dispatch has at
    // least 3 bytes of mask lookahead. `wbase > i` doubles as "invalid".
    let mut wbase = usize::MAX >> 1;
    let mut w = ClassWin::default();
    'outer: while i < n {
        if i + 64 > n {
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        }
        if wbase > i || i - wbase > 60 {
            // SAFETY: i + 64 <= n, checked above.
            w = unsafe { classify64(b.as_ptr().add(i)) };
            wbase = i;
        }
        let off = i - wbase; // <= 60
        let l = w.l >> off;
        let sp = w.sp >> off;
        // Pure lane. A position is impure if it is neither letter nor space,
        // or a space not followed by a letter inside the window (the shift
        // pulls in a zero at the top, so the window's last byte is
        // conservatively impure when it is a space).
        let impure = !(l | sp) | (sp & !(l >> 1));
        if impure & 1 == 0 {
            let region = impure.trailing_zeros() as usize; // >= 1
            let region_mask = 1u64.checked_shl(region as u32).map_or(u64::MAX, |v| v - 1);
            // Piece starts: the cursor plus every space in the region.
            let mut sm = (sp & region_mask) | 1;
            // Emit pieces between consecutive starts; the final start's piece
            // may extend past the region, so defer it to the dispatch path.
            loop {
                let rest = sm & (sm - 1);
                if rest == 0 {
                    let s0 = sm.trailing_zeros() as usize;
                    if s0 == 0 {
                        break; // cursor piece crosses the region: dispatch
                    }
                    i += s0;
                    continue 'outer;
                }
                f(
                    i + sm.trailing_zeros() as usize,
                    i + rest.trailing_zeros() as usize,
                );
                sm = rest;
            }
        }
        // Alt 2 fast path: \p{L}+ starting with an ASCII letter.
        if l & 1 != 0 {
            let run = (!l).trailing_zeros() as usize;
            let mut e = i + run;
            if off + run == 64 || (w.na >> (off + run)) & 1 != 0 {
                e = scan_letters(s, e);
            }
            f(i, e);
            i = e;
            continue;
        }
        // Alt 1 (contractions) must precede the prefix-word path.
        if b[i] == b'\'' {
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        }
        // Alt 2 with prefix: [^\r\n\p{L}\p{N}]?+ then ASCII letters.
        let pfx = !(w.l | w.d | w.nl | w.na); // prefix class (incl. space/tab)
        if (pfx >> off) & 1 != 0 && (l >> 1) & 1 != 0 {
            let run = (!(l >> 2)).trailing_zeros() as usize;
            let end_off = off + 2 + run;
            let mut e = i + 2 + run;
            if end_off == 64 || (w.na >> end_off) & 1 != 0 {
                e = scan_letters(s, e);
            }
            f(i, e);
            i = e;
            continue;
        }
        // Alt 3: \p{N}{1,3}, all-ASCII case.
        let d = w.d >> off;
        if d & 1 != 0 {
            let dr = (!d).trailing_zeros() as usize;
            if dr >= 3 {
                f(i, i + 3);
                i += 3;
                continue;
            }
            // dr in {1,2}: genuine non-digit inside the window ends the run,
            // unless it is non-ASCII (could be a Unicode digit) -> scalar.
            if (w.na >> (off + dr)) & 1 == 0 {
                f(i, i + dr);
                i += dr;
                continue;
            }
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        }
        // Alt 4: ` ?[^\s\p{L}\p{N}]++[\r\n]*`, all-ASCII case.
        let pu = !(w.l | w.d | w.ws | w.na); // strict ASCII punct
        let (pstart_off, pstart) = if (pu >> off) & 1 != 0 {
            (off, i)
        } else if b[i] == b' ' && (pu >> (off + 1)) & 1 != 0 {
            (off + 1, i + 1)
        } else {
            // Alts 5-7 (whitespace), all-ASCII case: a \s run containing a
            // newline ends after its last \r/\n; otherwise the final \s char
            // glues onto the next piece (`\s+(?!\S)`), except a length-1 run.
            let wsv = w.ws >> off;
            if wsv & 1 != 0 {
                let wr = (!wsv).trailing_zeros() as usize;
                if off + wr < 64 && (w.na >> (off + wr)) & 1 == 0 {
                    let nlrun = (w.nl >> off) & ((1u64 << wr) - 1);
                    let e = if nlrun != 0 {
                        i + 64 - nlrun.leading_zeros() as usize
                    } else {
                        i + (wr - 1).max(1)
                    };
                    f(i, e);
                    i = e;
                    continue;
                }
            }
            // Non-ASCII or window-boundary cases: scalar.
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        };
        let run = (!(pu >> pstart_off)).trailing_zeros() as usize;
        let end_off = pstart_off + run;
        if end_off == 64 || (w.na >> end_off) & 1 != 0 {
            // Run reaches the window end or a non-ASCII char: rescan exact.
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        }
        let tail = (!(w.nl >> end_off)).trailing_zeros() as usize;
        if end_off + tail == 64 {
            let e = next_piece_cl100k(s, i);
            f(i, e);
            i = e;
            continue;
        }
        let e = pstart + run + tail;
        f(i, e);
        i = e;
    }
}

/// Split `s` into pretokenization pieces, invoking `f(start, end)` per piece.
#[inline]
pub fn for_each_piece(s: &str, pattern: Pattern, mut f: impl FnMut(usize, usize)) {
    let mut i = 0;
    let n = s.len();
    match pattern {
        Pattern::Cl100k => {
            #[cfg(target_arch = "aarch64")]
            {
                for_each_piece_cl100k_masked(s, f);
            }
            #[cfg(not(target_arch = "aarch64"))]
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
