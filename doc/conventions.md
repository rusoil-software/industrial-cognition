# Development Conventions (KISS Principle Adherence)

These are the mandatory, high-level guidelines for any development task, ensuring code quality remains high while maintaining simplicity (KISS). These principles govern how code agents must operate and are derived from the system's required functionality and industry best practices.

1. **Domain Isolation (Separation of Concerns):** Every major subsystem (`camera`, `vision`, `detection`, `system`) **must** reside in its own isolated directory/module. Dependencies between modules must only flow through well-defined, abstract interfaces (APIs/message queues), never through direct state access.
2. **Single Responsibility Principle (SRP):** Every class, function, and module must have only one job. If a component handles both *sending* a message *and* *processing* the response, it must be split.
3. **Communication Protocol Authority:**
    * **Robot Control:** All physical control and data exchange with robotic manipulators must use **binary Modbus over RS-485** physical layer. This is the single source of truth for physical interaction protocols.
    * **Real-Time Data:** Use **WebSocket/FastAPI** for real-time streaming of detection results.
    * **Configuration/Status:** Use standard **REST** endpoints for querying state.
4. **Test Rigor (TDD Mandate):** Every piece of core business logic (in the `service.py` files) **must** have corresponding unit and integration tests before it is considered complete.
5. **Failure & Robustness:**
    * **System Failure:** Must handle I/O errors (network loss, DB disconnects) by reporting specific, actionable status codes (e.g., `COMM_ERROR`).
    * **Protocol Failure:** Modbus exceptions must be caught and translated into a standardized Python exception class, never allowing raw bus errors to propagate.

**Development Mandates:**

* **KISS Principle:** Prioritize the simplest working mechanism that meets the functional requirement before considering advanced patterns.
* **Documentation First:** New features must first result in a documentation update (in `doc/` files) before code changes are implemented.
