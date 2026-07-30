# Project Workflow Guide

This document dictates the mandatory process for feature development. **Process discipline is more important than raw code.**

## Development Workflow Sequence (Mandatory Order)

Every enhancement or feature addition **must** pass through these sequential stages:

1.  **Discovery/Specification:**
    *   **Goal:** Identify the *What* (Feature requirement/Bug).
    *   **Artifacts:** Update `doc/tasklist.md` (Mark task as '🟡 In Progress').
    *   **Documentation:** Create or update the necessary documentation in `doc/` files (e.g., `doc/new-feature.md`) to detail the vision for the change.
2.  **Design & Convention Check:**
    *   **Review:** Compare the proposed change against the rules defined in `doc/conventions.md`.
    *   **Refinement:** Update `doc/vision.md` and `doc/idea.md` if the change alters the core architecture or communication protocols.
3.  **Implementation:**
    *   **Backend Logic:** Write the service logic in the relevant domain directory (`src/<domain>/service.py`).
    *   **APIs:** Expose endpoints in the appropriate router (`src/<domain>/router.py`).
    *   **Testing First:** Implement the feature using Test-Driven Development (TDD). Write the unit tests first, then the minimum necessary code to make them pass.
4.  **Review & Commit:**
    *   **Code Review:** Submit the code for peer review, specifically pointing out the changes made to the development files.
    *   **Commit Message:** Commit message must reference the updated task list item ID.
5.  **Finalization:**
    *   Update `doc/tasklist.md` (Mark task as '✅ Completed').
    *   If required, create an `onboarding.md` guide update to reflect the new feature.