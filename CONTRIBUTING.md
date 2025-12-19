# Contribution Guidelines

Thank you for your interest in contributing to the CFGAN project! This guide will help you understand how to participate in the project development, including contribution processes, code standards, branch management, and more.

## Table of Contents

- [Contribution Workflow](#contribution-workflow)
- [Environment Setup](#environment-setup)
- [Code Standards](#code-standards)
- [Commit Message Format](#commit-message-format)
- [Branch Management Strategy](#branch-management-strategy)
- [Issue Reporting](#issue-reporting)
- [Pull Request](#pull-request)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [Code of Conduct](#code-of-conduct)

## Contribution Workflow

1. Search for [existing issues](https://github.com/zhoux77899/CFGAN/issues) to confirm that the problem or feature you want to address hasn't been handled yet
2. If it's a new issue or feature, create a new Issue to describe it
3. Fork the repository to your personal account
4. Create a new branch for development
5. Write code and ensure it passes all tests
6. Commit your changes and push to your Fork
7. Create a Pull Request and wait for project maintainers to review

## Environment Setup

### Installing Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/zhoux77899/CFGAN.git
   cd CFGAN
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

Pre-commit hooks will check code style and quality before each commit.

## Code Standards

The CFGAN project follows these code standards:

### Python Code Standards

- Use Python 3.10 or higher
- Follow the [PEP 8](https://pep8.org/) code style guide
- Use 4 spaces for indentation, not tabs
- Line width limit is 100 characters
- Use the isort tool to sort import statements (configured in pre-commit)
- Avoid using debug statements (pre-commit will check and warn about debug statements)
- Files must end with a newline character (pre-commit will handle this automatically)
- Avoid adding whitespace at the end of lines (pre-commit will handle this automatically)

### Documentation Standards

- Add docstrings to all public functions and classes, explaining their functionality, parameters, and return values
- Use Markdown format for documentation
- Comments in code should be in Chinese or English, maintaining consistency

## Commit Message Format

To keep the commit history clear and consistent, please follow this commit message format:

```
type(optional scope): short description

detailed description (if necessary)

Related Issue #(if applicable)
```

### Common Commit Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (whitespace, formatting, etc.)
- **refactor**: Code changes that neither fix a bug nor add a feature
- **perf**: Code changes that improve performance
- **test**: Adding or correcting tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts

### Commit Message Example

```
fix(config): Fix configuration file path parsing error

Fixed the logic error in handling os.PathLike type in get_value function

Related #12
```

## Branch Management Strategy

The CFGAN project adopts the following branch management strategy:

- **main**: Main branch containing the latest stable version code
- **develop**: Development branch for integrating new features and fixes
- **feature/<feature-name>**: Feature branches for developing specific features
- **fix/<issue-name>**: Fix branches for fixing specific bugs

### Branch Creation and Merging Process

1. Create a new feature or fix branch from the develop branch
2. Complete development on the branch and pass all tests
3. Push the code to your Fork
4. Create a Pull Request to the develop branch
5. Wait for code review and address all comments
6. After code review approval, maintainers will merge your branch

## Issue Reporting

If you find a bug or have a new feature suggestion, please create a new Issue on the [GitHub Issues page](https://github.com/zhoux77899/CFGAN/issues). When creating an Issue, please describe the problem or suggestion in as much detail as possible and provide information according to the following templates:

### Bug Report Template

```markdown
## Problem Description
Please clearly and concisely describe the problem you encountered.

## Reproduction Steps
1. What operation did you perform
2. What error occurred
3. What was the expected result

## Environment Information
- Operating System: 
- Python Version: 
- CFGAN Version: 
- Other Relevant Dependency Versions: 

## Error Logs
If there are error logs, please provide them here.

## Possible Solutions
If you have any possible solutions or suggestions, please provide them here.

## Additional Information
Any other information that might help solve the problem.
```

### Feature Request Template

```markdown
## Feature Description
Please clearly and concisely describe the feature you would like to add.

## Feature Value
How will this feature help the CFGAN project and users?

## Implementation Ideas
If you have any ideas about how to implement this feature, please provide them here.

## Additional Information
Any other relevant information or reference materials.
```

## Pull Request

When you have completed code development and are ready to submit, please create a Pull Request and fill in the information according to the following requirements:

### Pull Request Template

```markdown
## Related Issues
Please reference related Issue numbers here, e.g., Fixes #123

## Changes
Please briefly describe the changes you made.

## Implementation Details
Please describe your implementation methods and reasons in detail.

## Testing
Please describe what tests you performed and what the results were.

## Impact Scope
Which parts of the code or functionality might be affected by your changes?

## Additional Information
Any other information that might help with the review.
```

### Pull Request Notes

- Ensure your code passes all tests
- Ensure your code complies with the project's code standards
- Ensure your commit messages follow the correct format
- Ensure your Pull Request targets the correct branch (usually develop)
- Try to keep each Pull Request focused on a small, specific set of changes

## Testing

The CFGAN project uses Python's unittest framework for testing. Before submitting code, please ensure:

1. Run all tests in the project:
   ```bash
   python -m unittest discover tests
   ```

2. Add appropriate test cases for your new features or fixes

3. Ensure test coverage is as high as possible

## Documentation

Good documentation is crucial for open-source projects. Please ensure:

1. Add appropriate docstrings to your code
2. Update related README files and usage examples
3. Add new documentation pages if necessary

## Release Process

The release process for the CFGAN project is managed by the project maintainers. If you want to know about the current version or future version plans, please check the project's [GitHub page](https://github.com/zhoux77899/CFGAN).

## Code of Conduct

Please respect all contributors and follow basic open-source community etiquette. We hope to create an inclusive and friendly environment where everyone can comfortably participate in project development.

## Contact Information

If you have any questions or suggestions, please contact the project maintainers through:

- GitHub: https://github.com/zhoux77899/CFGAN
- Author: zhoux77899

Thank you again for your contribution to the CFGAN project!