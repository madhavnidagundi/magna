# Supply Chain Security

As a critical component for edge AI inference running on embedded hardware, Magna prioritizes supply chain security to ensure the integrity of inference results and protect edge devices from attacks.

## Automated Security Quality Gates

Our CI/CD pipeline enforces security policies on every Pull Request and via daily scheduled scans using the following tools:

### `cargo-audit`
We scan our dependency tree (`Cargo.lock`) against the RustSec Advisory Database. This ensures that no known security vulnerabilities (CVEs) are introduced into the project.

### `cargo-deny`
We enforce strict project policies around dependencies using `cargo-deny`. The configuration can be found in [`middleware/deny.toml`](middleware/deny.toml).
Our policies include:
- **Licenses**: We explicitly deny incompatible copyleft licenses (like GPL-3.0 and AGPL-3.0) and enforce the use of common open-source licenses.
- **Sources**: We enforce that all dependencies come from trusted registries (crates.io). We explicitly deny dependencies from unknown git repositories.
- **Advisories**: We fail builds on security vulnerabilities and warn on unmaintained or unsound crates.

## Manual Reviews
While we rely heavily on automated scanning, all new dependencies must still undergo a manual review during the Pull Request process to ensure they are strictly necessary and well-maintained.

If you find a security vulnerability, please refer to the security reporting guidelines (typically found in a SECURITY.md file or via private disclosure).
