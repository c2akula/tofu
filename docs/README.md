# Tofu Documentation

This directory contains the mdBook-based documentation for Tofu v1.0.0.

## 📚 Live Documentation

**The documentation is deployed at: https://c2akula.github.io/tofu/**

The site is automatically updated on every push to `develop` or `master` branches via GitHub Actions.

## Local Development

### Prerequisites

Install mdBook:
```bash
# Via cargo
cargo install mdbook

# Or via Homebrew (macOS)
brew install mdbook
```

### Building the Documentation

```bash
cd docs
mdbook build
```

The built documentation will be in `docs/book/`.

### Live Preview

To serve the documentation locally with live reload:
```bash
cd docs
mdbook serve
```

Then open http://localhost:3000 in your browser.

## Structure

- `book.toml` - mdBook configuration
- `src/` - Markdown source files
- `src/SUMMARY.md` - Table of contents
- `book/` - Generated HTML output (not committed to git)

## Deployment

Documentation is automatically deployed to GitHub Pages via the workflow in `.github/workflows/deploy-docs.yml`. Any push to `develop` or `master` branches triggers a new deployment.
