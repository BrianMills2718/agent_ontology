# Toolformer-Style Agent

The Toolformer agent augments a given text by detecting positions where tool calls would improve accuracy or add information, executing those calls, and selectively incorporating results back into the text. Rather than requiring explicit user instructions about which tools to use, the agent itself decides where tools are helpful.

The system accepts a TextInput containing the original text and an optional list of enabled tools. Three tools are available: a calculator for mathematical expressions, a search engine for factual lookups, and a QA model for answering specific questions extracted from context.

Processing begins at the detection step. A tool detector agent analyzes the input text and identifies candidate insertion points -- specific positions in the text where a tool call could add value. For each candidate, the detector outputs the position (character offset), the tool to call (calculator, search, or QA), the proposed tool input (the query or expression), and a brief justification for why the call would help. The detector produces a list of all candidates in a single call, outputting a CandidateList.

Next, the system fans out to execute all candidate tool calls in parallel. Each candidate is dispatched to its corresponding tool: calculator candidates go to the calculator tool, search candidates to the search tool, and QA candidates to the QA tool. Each returns a ToolResult containing the tool output and a success flag. This is a true parallel fan-out, since the tool calls are independent.

Once all tool results are collected, a filtering step applies a quality gate. A filter agent receives each candidate paired with its tool result and the surrounding text context, and decides whether incorporating the tool output would genuinely improve the text. It assigns an accept or reject decision along with a confidence score (0 to 1). Only candidates with an accept decision and confidence above 0.5 pass through the gate.

Finally, a text integrator agent receives the original text and the accepted tool results with their insertion positions. It produces the augmented text by weaving tool outputs into the original at the appropriate positions, adjusting for readability. The integrator also outputs an audit log listing which tools were called, which were accepted, and which were filtered out.

The output is a ToolformerOutput containing the augmented text, the audit log as a list of entries, and counts of candidates proposed, accepted, and rejected.
