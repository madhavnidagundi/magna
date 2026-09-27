# Student Project Roadmap

This directory contains design briefs for extending MISAL. The briefs define the problem space, desired outcomes, important constraints and evidence expected from the project. They are not fixed implementation specifications.

## Phase index

| Phase | Brief | Depends on | Earliest useful start |
|---|---|---|---|
| 1 | [Middleware refactoring](001_middleware-refactoring.md) | Team onboarding and a green portable build | First implementation phase |
| 2 | [Middleware hardening](002_middleware-hardening.md) | Stable Phase 1 tensor, capability and lifecycle contracts | After the affected Phase 1 contracts are reviewed |
| 3 | [OpenVINO CPU backend](003-extend-CPU-backend.md) | Stable backend trait, tensor validation and lifecycle contracts | Investigation may start early; implementation follows the relevant Phase 1/2 gates |
| 4 | [ROS 2 vision pipeline](004_vision-pipeline-integration.md) | Model metadata and a hardened gRPC tensor path | Contract and fixture work may start while hardening finishes |
| 5 | [Vision-pipeline containerization](005-containerization.md) | A functionally validated Phase 4 file-based pipeline | After the non-containerized path works |
| 6 | [Reproducible middleware profiling](006_inference-pipeline-profiling.md) | A correct, smoke-tested backend under test | CPU harness work may start first; each hardware lane opens independently |

The phase numbers identify work packages, not eight-person assignments or a requirement to finish every phase in one university term.

## Student ownership

The Student Development Team owns the solution. For each phase, the team is expected to:

1. inspect the current implementation and verify that the brief's assumptions are still true;
2. identify viable alternatives and their trade-offs;
3. present a recommendation for team discussion;
4. record the agreed decision and rationale in a short architecture decision record (ADR) or equivalent design note;
5. divide the work into issues and assign owners;
6. implement, review and integrate the selected solution; and
7. demonstrate the outcome with tests, measurements or other appropriate evidence.

Students may change proposed types, file locations, package boundaries, implementation languages, libraries, defaults, sequencing and deployment topology when they can explain why the alternative better satisfies the phase outcome.

## Open ownership with clear accountability

Working areas intentionally remain open until the team has investigated them. They are not assigned to named people by this roadmap. At the start of a milestone:

1. publish the agreed work packages as visible issues, including outcome, dependencies, acceptance evidence and an initial complexity/risk label;
2. let team members volunteer for a **directly responsible owner** role and at least one reviewer role;
3. record the owner and reviewer before implementation starts, while keeping unclaimed work visibly open;
4. use pairs or small groups for risky FFI, unsafe Rust, unfamiliar runtime and hardware work;
5. rotate facilitation, review and demonstration duties so responsibility does not remain with the same experienced contributors; and
6. return an issue to the open pool explicitly when an owner cannot continue; do not leave ownership implicit.

Ownership means coordinating an issue through design, implementation, validation and handoff. It does not mean working alone or having unilateral architecture authority. The whole team reviews consequential interfaces; the issue owner records the resulting decision.

For a team of eight, use a small number of active work streams rather than eight isolated tasks. A practical starting limit is three concurrent implementation streams, normally staffed by two or three people each. The team may change this limit based on evidence, but should avoid having more active pull requests than it can review promptly.

### Lightweight rotating roles

At each milestone kickoff, the team selects people for these time-bounded roles:

- **milestone facilitator:** maintains the board, calls out dependencies and arranges reviews;
- **technical owner per active stream:** coordinates the stream and its acceptance evidence;
- **validation owner:** checks that tests and demonstrations prove the stated outcome; and
- **integration steward:** watches shared interfaces and keeps the main branch usable.

These roles are coordination responsibilities, not permanent seniority levels. One person may hold only one major coordination role at a time unless the team records why an exception is necessary.

### Mixed-experience collaboration

- Label work by risk and prerequisite knowledge, not by “junior” or “senior” status.
- Pair a contributor learning an area with someone able to review it, especially for FFI, unsafe code, lifecycle changes and hardware SDKs.
- Reserve investigation, tests, documentation, fixtures and tooling as meaningful ownership opportunities; do not treat them as leftover work.
- Review for learning: explain contract and safety concerns, and let the issue owner present the result.
- Escalate a blocker after two working days without a credible next experiment, or sooner when hardware access or safety is involved.

## How to interpret requirement language

The words in the phase briefs have the following intent:

- **Required outcome:** Statements concerning correctness, explicit failure behavior, truthful test results, reproducibility and the phase definition of done describe outcomes that must be satisfied or explicitly deferred with approval.
- **Design direction:** Code sketches, proposed type names, file lists, numeric defaults, tools and implementation sequences are starting points. The team should challenge and refine them.
- **Example:** Wording such as “for example,” “such as,” “preferred,” or “suggested” is illustrative and may be replaced without special approval.
- **Change by decision:** A team may replace a proposed design direction with an equivalent solution after documenting the trade-off and reaching agreement in design review.

A phase is not expected to be implemented mechanically. Review should assess the quality of the reasoning and evidence as well as the resulting code.

## Planning and review cadence

Before implementation, each phase should produce a short proposal containing:

- the problem being solved and current-system evidence;
- assumptions and open questions;
- at least two options for consequential design decisions;
- the recommended approach and rejected alternatives;
- risks, dependencies and required hardware;
- a work breakdown with owners and a realistic milestone;
- validation and demonstration plans; and
- items intentionally deferred.

The proposal must also name the current milestone, expected calendar window or review date, available team capacity and the infrastructure owner for each required hardware or service dependency. Estimates are forecasts to support scope decisions, not commitments imposed on individual contributors.

The proposal should be discussed by the team before major interfaces are committed. Small reversible decisions do not require heavyweight documentation.

During implementation, decisions should be revisited when experiments contradict assumptions. At phase completion, the team should demonstrate the result and document remaining limitations rather than presenting partial or simulated behavior as complete.

Use a short ADR for consequential decisions. It should contain: context, considered options, decision, evidence, consequences, owner, review date and conditions that would trigger reconsideration. If consensus cannot be reached, record the competing options and ask the university supervisor or project sponsor to decide the unresolved constraint; do not hide disagreement in implementation.

## Roadmap relationships

```text
Middleware refactoring
        |
        v
Middleware hardening
        |
        +--------------------+
        |                    |
        v                    v
ROS 2 vision pipeline    Backend/runtime extensions
        |                    |
        v                    |
Containerization             |
        +---------+----------+
                  v
        Reproducible profiling
```

This is a dependency guide, not a mandatory calendar. Work may overlap when interfaces are stable and risks are understood. A small baseline measurement may be taken before refactoring, but comparative performance claims should use the hardened implementation.

Dependencies are gates on affected work, not reasons for the whole team to wait. Useful early work includes repository surveys, portable fixtures, model-profile design, runtime feasibility spikes and test-harness prototypes. Code that depends on an unstable shared interface should remain a short-lived experiment until that interface is accepted.

Before opening a hardware-dependent stream, identify who provides access, the supported SDK/runtime version and how absence will be reported. Missing hardware, vendor packages, models or permissions require replanning; they must not be converted into simulated completion.

## Scope management

All briefs may remain in the roadmap while the team negotiates depth per milestone. For every phase, classify work as:

- **Committed:** required for the current milestone;
- **Stretch:** attempted after committed outcomes are complete;
- **Deferred:** retained in the roadmap with a reason and prerequisite.

This keeps the full ambition visible without turning every idea into an unconditional deadline.

The candidate areas in each brief are a backlog of possible work, not a promise that every area is committed. For an initial milestone, prefer one thin, end-to-end, demonstrable path over partially starting every area. A scope change is valid when the team records the reason, impact and approval.

## Kickoff and completion gates

Before committing a phase or stream, confirm:

- its prerequisite interfaces are stable enough for the proposed work;
- required models, fixtures, SDKs, hardware and reviewers are available;
- acceptance criteria can run in the declared environment;
- an issue owner and reviewer have volunteered; and
- the work fits current capacity without abandoning active reviews.

A committed work area is complete only when:

1. its acceptance evidence passes in the declared portable or hardware environment;
2. required formatting, lint, test, documentation and dependency-policy checks pass;
3. behavior, limitations and consequential decisions are documented;
4. no required test reports success by silently skipping unavailable assets or hardware; and
5. the owner demonstrates the result and records follow-up, stretch and deferred work.

At milestone close, hold a short retrospective and ownership rotation. Record what was delivered, what remains open, active risks and which interfaces are safe for dependent phases.
