## system
You check exam questions against a source passage. Be strict. An option counts as supported only if the passage states it or it follows directly from what the passage states. Do not use outside knowledge to fill gaps in the passage.

For each option A to D, in order:
- evidence: the shortest quote from the passage that bears on this option, or "none".
- verdict: "supported" if the passage shows this option correctly answers the question; "contradicted" if the passage shows it does not; "not_in_passage" if the passage does not settle it.
- plausibility: for options other than the marked answer, how tempting it is to a student who skimmed the chapter: "low" (obviously wrong or unrelated to the topic), "medium" or "high". Use "n/a" for the marked answer.

Then:
- cognitive_level: "recall" if the answer is stated directly in the passage, "reasoning" if getting it needs an inference or applying a fact to a new case.
- stem_clear: true if the question has one clear reading and a student would know what is being asked.

Reply with JSON only.

## user
Passage:
"""
$passage
"""

Question: $stem
$options

Marked answer: $answer_letter
