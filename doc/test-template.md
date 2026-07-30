# Test Template

When writing a new test for a component, always follow this structure.

## 1. Test Identification
*   **Module/Service:** (Which component is being tested, e.g., `vision.service.object_detector`)
*   **Goal:** (What specific behavior must this test verify? e.g., "Detects object when confidence > 0.8.")
*   **Prerequisites:** (What must be set up first? e.g., "Mocked Modbus connection must be available.")

## 2. Test Implementation (TDD Cycle)
*   **Step 2a: Write Failing Test:** Write the test case *first*, assuming the code does not exist or is incorrect. The test must fail initially.
*   **Step 2b: Minimal Code Fix:** Write *only* the minimum amount of code required to make Test 2a pass. Do not add features, only add the necessary logic.
*   **Step 2c: Refactor/Expand:** Once the core test passes, improve readability, edge-case handling, or performance, ensuring all existing tests still pass.

## 3. Test Cases to Cover
Always test for the following states:
1.  **Happy Path:** Perfect input, expected output.
2.  **Boundary Conditions:** Min/Max values, zero values, empty inputs.
3.  **Failure Path:** Simulate system errors (e.g., connection loss, invalid data types, Modbus exception).