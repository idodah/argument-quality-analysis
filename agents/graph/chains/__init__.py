"""Pure LLM chains used by the refinement graph.

A chain is a stateless callable: prompt + model + parser. It takes primitive
inputs (strings) and returns primitive outputs (strings, enums, lists). Chains
do not read or mutate GraphState — that's the node layer's job.
"""