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
    NoStd,
    /// Comprehensive gate: fmt --check + clippy + deny.
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
        Cmd::NoStd => no_std(),
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

fn no_std() -> Result<(), String> {
    // Best-effort: ensure the bare-metal target is available locally.
    let _ = Command::new("rustup")
        .args(["target", "add", NO_STD_TARGET])
        .status();
    for (name, features) in NO_STD_CRATES {
        let mut args: Vec<String> = vec![
            "build".into(),
            "-p".into(),
            (*name).into(),
            "--no-default-features".into(),
            "--target".into(),
            NO_STD_TARGET.into(),
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
