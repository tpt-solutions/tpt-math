# tpt-math-units-dyn

Runtime dimension-checked units for values whose unit is only known when the
program runs — a YAML field that says `speed_limit: "90 km/h"`, a JSON payload
with `{"value": 12, "unit": "kWh"}`, a spreadsheet column header, a plugin
reporting whatever unit its sensor uses. It consolidates the earlier
`tpt-units-runtime` work into one crate and bridges it to the compile-time
quantities of `tpt-math-units`.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is the dynamic
half of the units layer: `tpt-math-units` (i.e. `uom`) checks dimensions at
compile time, this crate checks them at runtime, and the `interop` module
converts between the two so untrusted input can be validated once at the
boundary and stay statically typed everywhere after that. It is not an umbrella
crate; `tpt-math-units` is its only dependency.

## Features

| Feature | Enables | Notes |
|---|---|---|
| `ode` | `["dep:tpt-math-numeric"]` | Units-aware ODE integration (`solve::dopri5_dimensioned`). |
| `optimize` | `["dep:tpt-math-optimize-general", "dep:tpt-math-numeric"]` | Units-aware minimisation (`solve::minimize_dimensioned`). |
| `autodiff` | `["dep:tpt-math-autodiff-rev"]` | Units-aware gradients (`solve::gradient_dimensioned`). |

## Units-aware solvers

With `ode`, `optimize` or `autodiff` enabled, the `solve` module runs numerical
routines on dimensioned values. Every quantity is checked on the way in and out.

- `solve::dopri5_dimensioned` (`ode`) integrates `y′ = f(t, y)`.
- `solve::minimize_dimensioned` (`optimize`) runs L-BFGS on dimensioned
  parameters.
- `solve::gradient_dimensioned` (`autodiff`) attaches the dimension
  `cost / parameter` to a reverse-mode gradient.

Failures come back as `SolveError`: `Unit` for a dimension mismatch, `Ode` for
an integrator failure, or `Optimize` for an optimiser failure.

## Quick start

```toml
[dependencies]
tpt-math-units-dyn = "0.1"
```

```rust
use tpt_math_units_dyn::prelude::*;

// Values that arrived as (number, unit) pairs from a config file.
let distance = DynQuantity::parse(1.0, "km").unwrap();
let extra = DynQuantity::parse(200.0, "m").unwrap();

// Same dimension: addition is allowed, and conversion is implicit because
// every quantity is stored in SI base units.
let total = distance.try_add(&extra).unwrap();
assert_eq!(total.value(), 1200.0); // metres
assert!((total.convert_to("km").unwrap() - 1.2).abs() < 1e-12);

// Different dimensions: multiplication and division compose them ...
let duration = DynQuantity::parse(30.0, "min").unwrap();
let speed = total.try_div(&duration).unwrap();
assert_eq!(*speed.dim(), Dimension::VELOCITY);
assert!((speed.convert_to("km/h").unwrap() - 2.4).abs() < 1e-12);

// ... but addition across dimensions is rejected.
let mass = DynQuantity::parse(70.0, "kg").unwrap();
assert!(total.try_add(&mass).is_err());
```

Whole strings parse too, and the result can be handed to `uom` (add
`tpt-math-units = "0.1"` for this one):

```rust
use tpt_math_units::prelude::*;
use tpt_math_units::si::velocity::kilometer_per_hour;
use tpt_math_units_dyn::DynQuantity;

let dynamic: DynQuantity = "90 km/h".parse().unwrap();
let typed = Velocity::try_from(dynamic).unwrap();
assert!((typed.get::<kilometer_per_hour>() - 90.0).abs() < 1e-9);

let back = DynQuantity::from(typed);
assert!((back.convert_to("m/s").unwrap() - 25.0).abs() < 1e-9);
```

Application-specific units go in your own registry:

```rust
use tpt_math_units_dyn::{Dimension, UnitRegistry};

let mut registry = UnitRegistry::with_builtins();
registry.define("furlong", Dimension::LENGTH, 201.168);

let d = registry.quantity(2.0, "furlong").unwrap();
assert!((d.convert_to_with("m", &registry).unwrap() - 402.336).abs() < 1e-9);
```

The main types are `Dimension` (the seven SI base exponents as a `Copy` value
that multiplies, divides and compares like the quantity it describes),
`UnitDef` / `UnitRegistry` (symbol table mapping `"km"`, `"min"`, `"kWh"` to a
dimension plus the affine transform to SI base units), `DynQuantity` (a value
in SI base units tagged with its dimension), and `UnitError`. The free
functions `lookup`, `lookup_unit` and `convert` operate on the process-wide
`builtin_registry()`.

## Notes

- A `DynQuantity` always stores its magnitude in SI base units, so arithmetic
  needs no conversion bookkeeping and values built from different units of the
  same dimension compare directly.
- `Add`/`Sub` **panic** on a dimension mismatch, in the same spirit as an
  out-of-bounds index; use `try_add`/`try_sub` to get a `UnitError` instead.
  `Mul`/`Div` never mismatch — they compose dimensions — and only panic in the
  absurd case of an exponent overflowing `i8`. `Neg` is always infallible.
- Unit symbols are case sensitive, as SI requires: `"m"` is metre, `"T"` is
  tesla, `"t"` is tonne.
- Symbols are looked up whole. Named compound units such as `"km/h"` and `"kWh"`
  are entries in the 158-symbol built-in table (or in your own registry).
  `from_str` also accepts a product of base units such as `"m^2·kg·s^-2"`, which
  is what `Display` prints for compound units, so `Display` and `FromStr`
  round-trip.
- Degrees Celsius and Fahrenheit work through the affine `offset` field of
  `UnitDef` (`base = value * scale + offset`).
- Dimension exponents are `i8`; the `checked_mul`, `checked_div`,
  `checked_powi` and `checked_root` methods report overflow and non-integer
  roots instead of wrapping.
- `interop` bridges 18 `uom` quantities (`Length`, `Mass`, `Time`,
  `ElectricCurrent`, `ThermodynamicTemperature`, `AmountOfSubstance`,
  `LuminousIntensity`, `Area`, `Volume`, `Velocity`, `Acceleration`, `Force`,
  `Energy`, `Power`, `Pressure`, `Frequency`, `ElectricCharge`,
  `ElectricPotential`) via `From` and `TryFrom`.

## License

Licensed under either of MIT or Apache-2.0 at your option.
