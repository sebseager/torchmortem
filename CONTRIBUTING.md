# Contributing to examples

We want to make contributing to this project as easy and transparent as possible.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## Pull Requests

We actively welcome your pull requests.

If you're new, we encourage you to take a look at issues tagged with [good first issue](https://github.com/sebseager/torchmortem/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

If you don't see an issue that you'd like to work on, feel free to open a new one with a description of the problem you'd like to solve or feature you'd like to add. Please include clear motivation when proposing a new feature to allow for informed discussion.

Once you have an issue to work on, follow these steps:

0. Fork the repo and create your branch from `main`.
1. Make your code change.
2. Add tests for your change, if your code is not covered by existing tests. If your code cannot be demonstrated with an existing example in `examples/`, please add a new training script that exercises your change. 
3. Install `uv`. Install `torchmortem` into a virtual environment with something like:
    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    ```
4. Run all tests -- including any new ones you have written -- with `uv run pytest` and ensure they pass.
5. Python code in `torchmortem` uses the `ruff` formatter. Please format your code accordingly before submitting.

## License

By contributing to examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
