# Contributing to FiveForFree

Thank you for your interest in contributing to the NASDAQ Short-Term Volatility Prediction System!

## Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/teon-u/FiveForFree.git
cd FiveForFree
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up frontend**
```bash
cd frontend
npm install
cd ..
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY
```

## Code Standards

### Python Code
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Add docstrings to all classes and public methods
- Format code with Black: `black .`
- Lint with Flake8: `flake8 src/`

### JavaScript/React Code
- Follow ESLint configuration
- Use functional components with hooks
- Add PropTypes or TypeScript types
- Format with Prettier (if configured)

## Testing

### Run Python tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=src --cov-report=html
```

### Run frontend tests
```bash
cd frontend
npm test
```

## Commit Guidelines

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```bash
git commit -m "feat: add real-time prediction WebSocket endpoint"
```

## Pull Request Process

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes and commit them
3. Push to your fork: `git push origin feature/your-feature-name`
4. Create a Pull Request on GitHub
5. Wait for review and address any feedback

## Questions?

Open an issue on GitHub or contact the maintainers.
