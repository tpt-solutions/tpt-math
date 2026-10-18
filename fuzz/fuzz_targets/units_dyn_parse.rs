//! Fuzz the `DynQuantity` text parser. Arbitrary input must never panic. When a
//! string parses, formatting it and parsing again must give the same quantity.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tpt_math_units_dyn::DynQuantity;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    if let Ok(q) = text.parse::<DynQuantity>() {
        let shown = q.to_string();
        let again: DynQuantity = shown
            .parse()
            .expect("a formatted quantity must parse back");
        assert_eq!(q, again, "round trip through {shown:?}");
    }
});
