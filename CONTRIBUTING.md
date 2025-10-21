# Contributing to PepTron

We welcome contributions! Here's how to get started:

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/peptron.git
   cd peptron
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
# Build and run Docker container (recommended)
docker build -t peptron:latest .
docker run --gpus all -it --rm -v $(pwd):/workspace peptron:latest
```

## Making Changes

1. **Write clean code** following Python best practices
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Test your changes**:

## Submit Your Contribution

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear description of what you changed
   - Why the change is needed
   - Any testing you performed

## Contribution Types

- **🐛 Bug fixes**: Fix errors, memory issues, or compatibility problems
- **✨ New features**: Add model improvements, data formats, or utilities
- **📚 Documentation**: Improve README, add examples, or fix typos
- **🚀 Performance**: Optimize speed, memory usage, or GPU utilization

## Code Style

- Use Python 3.8+ with type hints
- Follow PEP 8 style guidelines
- Keep lines under 88 characters
- Add docstrings to functions and classes

## Need Help?

- Check existing [Issues](https://github.com/PeptoneLtd/peptron/issues) first
- Open a new issue for bugs or feature requests
- Be respectful and patient with reviewers

Thanks for contributing to protein structure prediction research! 🧬
