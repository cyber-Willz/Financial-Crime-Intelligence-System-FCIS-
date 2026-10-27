//! Helpers for the SAML-D (Synthetic Anti-Money Laundering Dataset) CSV format.

pub fn country_iso2(name: &str) -> Option<&'static str> {
    Some(match name.trim() {
        "UK" => "GB",
        "USA" => "US",
        "UAE" => "AE",
        "Mexico" => "MX",
        "Turkey" => "TR",
        "Morocco" => "MA",
        "Nigeria" => "NG",
        "Albania" => "AL",
        "Pakistan" => "PK",
        "India" => "IN",
        "Italy" => "IT",
        "Germany" => "DE",
        "Austria" => "AT",
        "Japan" => "JP",
        "Switzerland" => "CH",
        "France" => "FR",
        "Spain" => "ES",
        "Netherlands" => "NL",
        "United Kingdom" => "GB",
        "Afghanistan" => "AF",
        "Iran" => "IR",
        "North Korea" => "KP",
        "Myanmar" => "MM",
        "Syria" => "SY",
        "Malaysia" => "MY",
        "Qatar" => "QA",
        "Saudi Arabia" => "SA",
        _ => return None,
    })
}

pub fn currency_iso(name: &str) -> Option<&'static str> {
    Some(match name.trim() {
        "UK pounds" => "GBP",
        "Euro" => "EUR",
        "Moroccan dirham" => "MAD",
        "Dirham" => "AED",
        "Swiss franc" => "CHF",
        "Turkish lira" => "TRY",
        "Indian rupee" => "INR",
        "Mexican Peso" => "MXN",
        "Yen" => "JPY",
        "Albanian lek" => "ALL",
        "Pakistani rupee" => "PKR",
        "Naira" => "NGN",
        "US dollar" => "USD",
        _ => return None,
    })
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parse `YYYY-MM-DD` + `HH:MM:SS` (UTC) into unix seconds.
pub fn parse_ts(date: &str, time: &str) -> Option<i64> {
    let mut d = date.split('-');
    let (y, m, dd) = (
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
    );
    let mut t = time.split(':');
    let (h, mi, s) = (
        t.next()?.parse::<i64>().ok()?,
        t.next()?.parse::<i64>().ok()?,
        t.next().unwrap_or("0").parse::<i64>().ok()?,
    );
    Some(days_from_civil(y, m, dd) * 86_400 + h * 3600 + mi * 60 + s)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ts_epoch() {
        assert_eq!(parse_ts("1970-01-01", "00:00:00"), Some(0));
        assert_eq!(parse_ts("2022-10-07", "10:35:19"), Some(1_665_138_919));
    }
}
