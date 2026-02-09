# Panel Review Agent

The Panel Review agent evaluates a proposal document by dispatching it to three independent specialist reviewer agents in parallel, then synthesizing their feedback through a moderator agent that produces a weighted final recommendation. This architecture ensures that proposals are assessed from multiple expert perspectives before a decision is made.

The system accepts a ProposalInput containing the proposal text, the proposal title, and the author name. Processing begins at an intake step that validates the proposal is non-empty, assigns a unique review ID, and stores the proposal in a review store (key-value, persistent) for audit purposes.

From intake, the system fans out to three parallel reviewer agents. The technical reviewer agent evaluates the proposal for technical feasibility, architecture soundness, scalability risks, and implementation complexity. It produces a ReviewOutput containing a score (1 to 10), a list of strengths, a list of concerns, and a narrative assessment. The business reviewer agent evaluates market fit, cost-benefit, competitive positioning, and revenue impact, producing the same ReviewOutput structure. The UX reviewer agent evaluates user experience implications, accessibility, usability friction, and design coherence, also producing a ReviewOutput.

All three reviewers operate independently and in parallel. They share no state and receive identical copies of the proposal. Each reviewer's step invokes its corresponding agent and collects the structured output.

Once all three reviews are complete, a synthesis step gathers the three ReviewOutputs. The moderator agent receives all three reviews along with a weight configuration: technical is weighted at 0.4, business at 0.35, and UX at 0.25. The moderator computes a weighted composite score, identifies consensus points (issues raised by two or more reviewers), flags conflicts (where reviewers disagree), and produces a final recommendation of approve, revise, or reject.

Before emitting the final output, a quality gate checks whether any individual reviewer score is below 3. If so, the proposal is automatically flagged as requiring major revision regardless of the composite score, and the recommendation is overridden to revise. This gate ensures that a critical failure in any one dimension cannot be masked by high scores in others.

The output is a PanelReviewResult containing the composite score, the final recommendation, a summary narrative, the list of consensus points, the list of conflicts, and the three individual reviews for transparency. The result along with the original proposal are persisted to the review store.
