//! `xtask` — developer tooling for the `tpt-math` workspace.
//!
//! Provides the one-stop `cargo xtask <cmd>` entry points used both locally and
//! in CI. Every command shells out to `cargo` (and, where relevant, `rustup` /
//! `cargo-deny`) so the behaviour is transparent and easy to reproduce by hand.
//!
//! The `no-std` command builds the workspace's `no_std = true` crates for
//! `thumbv6m-none-eabi`. The crate list mirrors the `no_std = true` entries for
//! this repo in `../tpt-rust-map/registry.toml`; keep the two in sync.

use std::process::Command;

use clap::{Parser, Subcommand};

/// Workspace maintenance tasks.
#[derive(Parser)]
#[command(name = "xtask", about = "tpt-math workspace tooling", version)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run `cargo fmt --all`.
    Fmt,
    /// Run workspace clippy with `-D warnings`.
    Clippy,
    /// Run the full workspace test suite.
    Test,
    /// Run `cargo deny check`.
    Deny,
    /// Build the `no_std = true` crates for `thumbv6m-none-eabi`.
    NoStd {
        /// Target triple to build for. Defaults to the bare-metal `thumbv6m-none-eabi`.
        #[arg(long)]
        target: Option<String>,
    },
    /// Check that the no_std list, the crates declaring `#![no_std]`, and the
    /// README `no_std` column agree.
    Drift,
    /// Full local gate, matching the CI jobs: fmt --check, clippy, tests,
    /// doctests, no_std build, drift check and cargo-deny.
    Check,
}

/// `(crate name, extra no_std features)`. The features enable the `alloc` /
/// `libm` options the crate needs to compile for a bare-metal target.
const NO_STD_CRATES: &[(&str, &str)] = &[
    ("tpt-math-numeric", "libm,alloc"),
    ("tpt-math-exact", "alloc"),
    ("tpt-math-units", "alloc"),
    ("tpt-math-linalg", "alloc"),
    ("tpt-math-prob-core", "alloc"),
    ("tpt-math-prob-dist", "alloc"),
    ("tpt-math-prob-sampler", "alloc"),
    ("tpt-math-autodiff-fwd", "alloc"),
    ("tpt-math-linalg-dense", "alloc"),
    ("tpt-math-linalg-sparse", "alloc"),
    ("tpt-math-linalg-complex", "alloc"),
    ("tpt-math-linalg-fixed", ""),
    ("tpt-math-geometry", ""),
    ("tpt-math-spatial", ""),
    ("tpt-math-graph", ""),
    ("tpt-math-interpolate", "alloc"),
    ("tpt-math-pde-operators", "alloc"),
];

const NO_STD_TARGET: &str = "thumbv6m-none-eabi";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Fmt => cargo(&["fmt", "--all"]),
        Cmd::Clippy => cargo(&[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ]),
        Cmd::Test => cargo(&["test", "--workspace", "--all-features"]),
        Cmd::Deny => cargo_deny(&["check"]),
        Cmd::NoStd { target } => no_std(target.as_deref().unwrap_or(NO_STD_TARGET)),
        Cmd::Drift => drift(),
        Cmd::Check => {
            cargo(&["fmt", "--all", "--", "--check"])?;
            cargo(&[
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ])?;
            cargo(&["test", "--workspace", "--all-features"])?;
            cargo(&["test", "--workspace", "--all-features", "--doc"])?;
            cargo_with_env(
                &["doc", "--workspace", "--no-deps", "--all-features"],
                &[("RUSTDOCFLAGS", "-D warnings")],
            )?;
            no_std(NO_STD_TARGET)?;
            drift()?;
            cargo_deny(&["check"])?;
            Ok(())
        }
    }
}

fn cargo(args: &[&str]) -> Result<(), String> {
    let status = Command::new("cargo")
        .args(args)
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo {}` failed", args.join(" ")))
    }
}

fn cargo_with_env(args: &[&str], envs: &[(&str, &str)]) -> Result<(), String> {
    let status = Command::new("cargo")
        .args(args)
        .envs(envs.iter().copied())
        .status()
        .map_err(|e| format!("failed to spawn cargo: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo {}` failed", args.join(" ")))
    }
}

fn cargo_deny(args: &[&str]) -> Result<(), String> {
    // `cargo-deny` may be installed either as the standalone `cargo-deny`
    // binary or as the cargo subcommand `cargo deny`.
    let has_standalone = which("cargo-deny");
    let mut cmd = Command::new(if has_standalone {
        "cargo-deny"
    } else {
        "cargo"
    });
    if !has_standalone {
        cmd.arg("deny");
    }
    cmd.args(args);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn cargo-deny: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`cargo deny {}` failed", args.join(" ")))
    }
}

/// Compare three views of the `no_std` crate set and fail on any difference:
///
/// 1. crates whose `src/lib.rs` declares `#![no_std]`,
/// 2. the `NO_STD_CRATES` list used by [`no_std`],
/// 3. README table rows whose `no_std` column starts with `yes`.
fn drift() -> Result<(), String> {
    use std::collections::BTreeSet;
    use std::fs;

    let mut declared = BTreeSet::new();
    let entries = fs::read_dir("crates").map_err(|e| format!("reading crates/: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading crates/: {e}"))?;
        let lib = entry.path().join("src").join("lib.rs");
        let Ok(source) = fs::read_to_string(&lib) else {
            continue;
        };
        if source.contains("#![no_std]") {
            declared.insert(entry.file_name().to_string_lossy().into_owned());
        }
    }

    let listed: BTreeSet<String> = NO_STD_CRATES
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();

    let readme = fs::read_to_string("README.md").map_err(|e| format!("reading README.md: {e}"))?;
    let mut documented = BTreeSet::new();
    for line in readme.lines().filter(|l| l.starts_with("| `tpt-math-")) {
        // Columns: "", crate, wraps, no_std, notes, "".
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if let (Some(name), Some(no_std)) = (cells.get(1), cells.get(3)) {
            if no_std.starts_with("yes") {
                documented.insert(name.trim_matches('`').to_string());
            }
        }
    }

    let mut problems = Vec::new();
    for (label, other) in [
        ("NO_STD_CRATES", &listed),
        ("the README no_std column", &documented),
    ] {
        for name in declared.difference(other) {
            problems.push(format!(
                "{name} declares #![no_std] but is missing from {label}"
            ));
        }
        for name in other.difference(&declared) {
            problems.push(format!(
                "{name} is listed in {label} but does not declare #![no_std]"
            ));
        }
    }
    if problems.is_empty() {
        println!(
            "no_std drift check: {} crates agree across all three sources",
            declared.len()
        );
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

fn no_std(target: &str) -> Result<(), String> {
    // Best-effort: ensure the bare-metal target is available locally.
    let _ = Command::new("rustup")
        .args(["target", "add", target])
        .status();
    for (name, features) in NO_STD_CRATES {
        let mut args: Vec<String> = vec![
            "build".into(),
            "-p".into(),
            (*name).into(),
            "--no-default-features".into(),
            "--target".into(),
            target.into(),
        ];
        if !features.is_empty() {
            args.push("--features".into());
            args.push((*features).into());
        }
        let status = Command::new("cargo")
            .args(&args)
            .status()
            .map_err(|e| format!("failed to spawn cargo: {e}"))?;
        if !status.success() {
            return Err(format!(
                "no_std build of `{name}` (features: `{features}`) failed"
            ));
        }
    }
    Ok(())
}

/// Cheap `Path::exists`-style probe for an executable on `PATH`.
fn which(name: &str) -> bool {
    Command::new(if cfg!(windows) { "where" } else { "which" })
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
