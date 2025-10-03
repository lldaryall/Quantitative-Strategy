# Contributing to qbacktester

Thank you for your interest in contributing to qbacktester! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a feature branch
5. Make your changes
6. Run tests and linting
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Make (optional, for using Makefile commands)

### Quick Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/qbacktester.git
cd qbacktester

# Run the development setup script
python scripts/setup_dev.py

# Or set up manually:
pip install -e .[dev]
pre-commit install
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import qbacktester; print(qbacktester.__version__)"
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code structure without changing functionality

### Before You Start

1. Check existing issues and pull requests to avoid duplicates
2. For large changes, open an issue first to discuss the approach
3. Ensure your changes align with the project's goals and architecture

## Code Style

### Python Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **ruff**: Additional linting

### Formatting

```bash
# Format code
make format
# or
black src/ tests/
isort src/ tests/
```

### Type Hints

All public functions and methods should have type hints:

```python
def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns for given periods."""
    return prices.pct_change(periods)
```

### Docstrings

Use Google-style docstrings for all public functions:

```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (default: 0.0)
        
    Returns:
        Sharpe ratio
        
    Raises:
        ValueError: If returns is empty or invalid
    """
```

### Naming Conventions

- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants
- Use descriptive names that clearly indicate purpose

## Testing

### Running Tests

```bash
# Run all tests
make test
# or
pytest

# Run with coverage
make test-cov
# or
pytest --cov=qbacktester --cov-report=html

# Run specific test file
pytest tests/test_backtester.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common test data

### Test Structure

```python
def test_calculate_returns_with_valid_data():
    """Test returns calculation with valid data."""
    prices = pd.Series([100, 105, 110, 108])
    expected = pd.Series([np.nan, 0.05, 0.0476, -0.0182])
    result = calculate_returns(prices)
    pd.testing.assert_series_equal(result, expected, atol=1e-4)


def test_calculate_returns_with_empty_series():
    """Test returns calculation with empty series."""
    with pytest.raises(ValueError, match="Empty series"):
        calculate_returns(pd.Series(dtype=float))
```

## Documentation

### Code Documentation

- Document all public APIs
- Include examples in docstrings where helpful
- Keep documentation up to date with code changes

### README Updates

- Update README.md for significant changes
- Include new features in the features list
- Update installation instructions if needed
- Add new examples or use cases

### Type Documentation

- Use type hints throughout
- Document complex types with comments
- Use `typing` module for complex type annotations

## Pull Request Process

### Before Submitting

1. **Run all checks**:
   ```bash
   make ci
   ```

2. **Ensure tests pass**:
   ```bash
   make test
   ```

3. **Check code quality**:
   ```bash
   make lint
   make type-check
   make security
   ```

4. **Update documentation** if needed

5. **Add changelog entry** for significant changes

### PR Description

Include the following in your PR description:

- **Summary**: Brief description of changes
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How you tested the changes
- **Breaking Changes**: Any breaking changes
- **Related Issues**: Link to related issues

### PR Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Breaking Changes
- [ ] No breaking changes
- [ ] Breaking changes documented

## Related Issues
Fixes #(issue number)
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Keep PR focused and reasonably sized
5. Update PR if requested

## Release Process

### Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Changelog

Update `CHANGELOG.md` with:

- New features
- Bug fixes
- Breaking changes
- Deprecations
- Security updates

### Release Steps

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Merge after review
5. Create GitHub release
6. Publish to PyPI (maintainers only)

## Development Tools

### Makefile Commands

```bash
make help          # Show all available commands
make install       # Install package
make install-dev   # Install with dev dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linting
make format        # Format code
make type-check    # Run type checking
make security      # Run security checks
make clean         # Clean build artifacts
make build         # Build package
make ci            # Run all CI checks
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit:

- Code formatting (Black, isort)
- Linting (flake8, ruff)
- Type checking (mypy)
- Security checks (bandit)
- File checks (trailing whitespace, etc.)

### IDE Configuration

#### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort
- GitLens

#### PyCharm

Recommended plugins:
- Black
- isort
- mypy

## Getting Help

- **Documentation**: Check README.md and docstrings
- **Issues**: Search existing issues or create new ones
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in PR comments

## License

By contributing to qbacktester, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to qbacktester! 🚀

