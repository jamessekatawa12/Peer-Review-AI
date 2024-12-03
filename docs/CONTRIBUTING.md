# Contributing to AI Multi-Agent Framework for Peer Review

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html)

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issue tracker](https://github.com/jamessekatawa12/peer_review_ai/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/jamessekatawa12/peer_review_ai/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Process

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

4. Make your changes and ensure all tests pass:
```bash
pytest tests/
```

## Code Style

- We use [Black](https://github.com/psf/black) for Python code formatting
- We use [isort](https://pycqa.github.io/isort/) for import sorting
- We use [flake8](https://flake8.pycqa.org/) for style guide enforcement
- We use [mypy](http://mypy-lang.org/) for static type checking

## Adding New Agents

1. Create a new file in the appropriate directory under `agents/`
2. Inherit from `BaseAgent` and implement required methods
3. Add tests in `tests/agents/`
4. Update the API to include your new agent
5. Document the agent's capabilities and usage

## Documentation

- Use docstrings for all public modules, functions, classes, and methods
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update README.md with any new features or changes
- Add examples to the documentation if appropriate

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 