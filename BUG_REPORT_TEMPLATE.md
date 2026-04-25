# Bug Report Template

## Summary

Briefly describe the issue you are reporting in one sentence (max 80 characters).

```

```bash
## Details
1. Reproduction Steps
  - List all steps to reproduce the bug, including any commands, input values, or user actions that cause the problem.
2. Expected Result
  - Describe what you expected to happen when following the reproduction steps.
3. Actual Result
  - Describe what actually happened instead of the expected result.
4. Screenshots/Logs (optional)
  - If applicable, include any relevant screenshots or logs that help explain the issue.
5. Environment Details
  - List the operating system, Python version, and other relevant environment details.

```

```markdown
## Additional Information (optional)

- Any workarounds you have found to bypass the bug.
- Links to related issues or pull requests.
- Any other information that may be helpful in diagnosing the issue.

## Labels (optional)

- If applicable, suggest labels for this issue based on its severity, priority, and component affected. For example:
- `bug`
- `high-priority`
- `backend`

```

The following example demonstrates how to fill out the bug report template for a specific issue:

```markdown
# Bug Report: Backend service crashes during test run

## Summary
Backend service crashes during test execution.

## Details
1. Reproduction Steps
  - Run the tests for the backend service (e.g., `pytest backend/tests`)
2. Expected Result
  - All tests should pass without any errors or exceptions.
3. Actual Result
  - The test suite crashes with an unhandled exception during execution.
4. Screenshots/Logs (optional)
  - Attach log files from the test run and terminal output where the crash occurred.
5. Environment Details
  - Operating System: Linux Ubuntu 20.04
  - Python Version: 3.8.10
  - Other Relevant Details (if applicable): FastAPI version, database connection details, etc.

## Additional Information (optional)
- The issue seems to be related to a specific test case in the backend service tests. Attached logs show the point of failure and stack trace.
- This issue was also reported by another user who ran into the same problem on their machine. Here's the link to their report: [Issue #123](https://github.com/petra-ai/backend/issues/123)

## Labels (optional)
- `bug`
- `high-priority`
- `backend`