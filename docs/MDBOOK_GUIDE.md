# mdBook Quick Reference

## Installation

**macOS:**
```bash
brew install mdbook
```

**Via Cargo (all platforms):**
```bash
cargo install mdbook
```

**Pre-built binaries:**
Download from https://github.com/rust-lang/mdBook/releases

## Common Commands

### Build the book
```bash
cd docs
mdbook build
```

### Serve with live reload (development)
```bash
cd docs
mdbook serve
# Opens at http://localhost:3000
```

### Serve on different port
```bash
mdbook serve -p 8080
```

### Clean build artifacts
```bash
mdbook clean
```

### Test code blocks
```bash
mdbook test
```

## Adding New Pages

1. Create the markdown file in `src/` directory
2. Add entry to `src/SUMMARY.md`
3. mdBook will automatically update the navigation

## Configuration

Edit `book.toml` to customize:
- Book metadata (title, authors)
- Output format settings
- Theme and styling
- Plugins and preprocessors

## Useful Links

- mdBook Documentation: https://rust-lang.github.io/mdBook/
- mdBook GitHub: https://github.com/rust-lang/mdBook
