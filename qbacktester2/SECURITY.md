# Security Policy

## Supported Versions

We provide security updates for the following versions of qbacktester:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in qbacktester, please follow these steps:

### 1. Do NOT create a public issue

**Do not** create a public GitHub issue for security vulnerabilities. This could potentially expose the vulnerability to malicious actors.

### 2. Report privately

Please report security vulnerabilities privately by:

- **Email**: Send details to `security@example.com` (replace with actual security contact)
- **GitHub Security Advisory**: Use GitHub's private vulnerability reporting feature if available

### 3. Include the following information

When reporting a vulnerability, please include:

- **Description**: Clear description of the vulnerability
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Impact**: Potential impact and severity assessment
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have ideas for fixing the issue
- **Your contact information**: For follow-up questions

### 4. Response timeline

We will:

- **Acknowledge** your report within 48 hours
- **Investigate** the vulnerability within 7 days
- **Provide updates** on our progress
- **Release a fix** as soon as possible (typically within 30 days)
- **Credit** you in our security advisories (unless you prefer to remain anonymous)

## Security Considerations

### Data Security

qbacktester handles financial data, so we take data security seriously:

- **Local caching**: Data is cached locally in Parquet format
- **No data transmission**: We don't transmit your data to external services
- **Secure defaults**: Sensitive operations require explicit configuration

### Network Security

- **HTTPS only**: All external API calls use HTTPS
- **Retry logic**: Built-in retry with exponential backoff
- **Timeout handling**: Configurable timeouts for network operations

### Code Security

- **Input validation**: All inputs are validated before processing
- **Error handling**: Comprehensive error handling to prevent information leakage
- **Dependency management**: Regular updates of dependencies
- **Security scanning**: Automated security scanning with bandit

## Security Best Practices

### For Users

1. **Keep updated**: Always use the latest version
2. **Secure data**: Store cached data in secure locations
3. **Environment variables**: Use environment variables for sensitive configuration
4. **Network security**: Use secure networks when downloading data
5. **Access control**: Limit access to your data directories

### For Developers

1. **Input validation**: Always validate user inputs
2. **Error handling**: Don't expose sensitive information in error messages
3. **Dependencies**: Keep dependencies updated
4. **Code review**: Security-focused code reviews
5. **Testing**: Include security testing in your workflow

## Security Tools

We use several tools to maintain security:

- **Bandit**: Static analysis for security issues
- **Safety**: Check for known security vulnerabilities in dependencies
- **Pre-commit hooks**: Automated security checks
- **Dependabot**: Automated dependency updates

## Vulnerability Disclosure

When we discover or receive reports of vulnerabilities:

1. **Assessment**: We assess the severity and impact
2. **Fix development**: We develop a fix as quickly as possible
3. **Testing**: We thoroughly test the fix
4. **Release**: We release a patched version
5. **Advisory**: We publish a security advisory
6. **Credit**: We credit the reporter (if desired)

## Security Advisories

Security advisories are published in:

- GitHub Security Advisories
- CHANGELOG.md
- Release notes

## Contact

For security-related questions or concerns:

- **Security Email**: `security@example.com`
- **General Issues**: Use GitHub Issues for non-security issues
- **Discussions**: Use GitHub Discussions for general questions

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities to us. Your efforts help make qbacktester more secure for everyone.

## Legal

This security policy is provided for informational purposes only. It does not create any legal obligations or warranties. Users are responsible for their own security practices and compliance with applicable laws and regulations.

---

**Last updated**: January 2025

