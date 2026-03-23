# Contributing to MIR System

Thank you for taking the time to contribute! 🎵

## Quick Setup

```bash
git clone https://github.com/yourusername/mir-system.git
cd mir-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
sudo apt-get install ffmpeg libsndfile1   # or brew install ffmpeg on macOS
cp .env.example .env
```

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code. Only merged via PR with passing CI. |
| `develop` | Integration branch. All feature branches merge here first. |
| `feature/*` | New features or non-trivial changes. |
| `fix/*` | Bug fixes. |
| `chore/*` | Refactoring, dependency updates, tooling. |

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(features): add spectral flux extraction
fix(search): handle empty FAISS index during search
chore(deps): bump librosa to 0.10.2
docs(readme): add GPU setup instructions
test(api): add batch upload integration test
```

## Code Style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- **Types**: `mypy` — all public functions must be typed

```bash
ruff format src/ api/ tests/
ruff check src/ api/ tests/
mypy src/ api/
```

## Testing

All new code must include tests. Aim for ≥90% coverage on new modules.

```bash
pytest tests/unit/ -v            # fast, no external services
pytest tests/integration/ -v     # requires Redis (see docker-compose.yml)
pytest tests/ --cov=src --cov=api --cov-report=html
```

## Pull Request Checklist

Before opening a PR, please confirm:

- [ ] Tests pass locally (`pytest tests/unit/`)
- [ ] Linting passes (`ruff check`)
- [ ] Type check passes (`mypy src/ api/`)
- [ ] PR description explains *what* and *why*
- [ ] Breaking changes are called out clearly

## Reporting Issues

Please include:
1. Python version (`python --version`)
2. OS and architecture
3. Full stack trace
4. Minimal reproduction steps

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). 
Be kind and constructive.
