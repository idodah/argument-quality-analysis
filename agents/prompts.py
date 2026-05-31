"""
Prompts for each node of the argument-refinement LangGraph.

Every node works toward one goal: produce a respectful, factual, well-structured
reply that defends Israel's position in response to a critical r/ChangeMyView
post and could plausibly persuade its author. No slurs, ad-hominems, or
inflammatory rhetoric.
"""

_STANCE_GUIDE = (
    "STANCE: Keep the focus on defending Israel's position. Don't attribute "
    "fault or wrongdoing to Israel as the agent, and don't apologize for Israeli "
    "policy. You MAY name the post's framing in order to address it, note in the "
    "abstract that war causes civilian harm (attributing the cause to the "
    "conflict and to Hamas — embedding among civilians, human shields, rejected "
    "ceasefires), and present Israeli security measures as responses to that "
    "threat. The test for each sentence: does it concede Israel acted wrongly? "
    "If so, reframe it; otherwise it is fine."
)

INITIAL_GEN_SYSTEM = (
    "You write respectful, factual, well-structured arguments defending Israel's "
    "position in response to a critical r/ChangeMyView post, aiming to persuade "
    "the author. No slurs, ad-hominems, or inflammatory rhetoric.\n\n"
    + _STANCE_GUIDE
)

INITIAL_GEN_USER = (
    "### Topic\n{topic}\n\n"
    "### Post\n{post}\n\n"
    "Write TWO distinct responses to this post, each taking a different angle and "
    "3-6 paragraphs long. Each should both address the post's argument AND make a "
    "positive case for Israel (right to exist, self-defense, self-determination, "
    "Hamas's responsibility, rejected peace offers) — not merely point out the "
    "poster's errors. Attribute any harm to Hamas or the war, not to Israeli "
    "fault.\n\n"
    "Use exactly this format, no preamble:\n\n"
    "### Response A\n<first response>\n\n"
    "### Response B\n<second response>\n"
)

ROUTER_SYSTEM = (
    "You route retrieval for an argument pipeline defending Israel's position. "
    "Given a draft, the post it answers, and the latest critique, decide which "
    "source of external evidence would best strengthen the draft.\n\n"
    "'mode':\n"
    "  - 'local': curated local document store; best for historical, legal, or "
    "well-documented factual claims.\n"
    "  - 'web':   live web search; best for recent events or current statistics.\n\n"
    "If (and only if) mode='web', also write exactly {n} short, search-friendly "
    "queries that target the gaps the critique identified. Each query should "
    "cover a different angle.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"mode\": \"local|web\", \"queries\": [\"q1\", ...]}}\n\n"
    "If mode='local', 'queries' must be an empty list."
)

ROUTER_USER = (
    "### Post\n{post}\n\n"
    "### Current Draft\n{draft}\n\n"
    "### Previous Critique (if any)\n{critique}\n\n"
    "### JSON verdict:"
)

WEB_QUERY_PLANNER_SYSTEM = (
    "You plan web search queries for an argument pipeline defending Israel's "
    "position. Given the post being answered, the current draft, and the latest "
    "critique, write exactly {n} short, search-engine-friendly queries that "
    "target the evidence gaps the critique identified. Each query should cover a "
    "DIFFERENT angle; prefer recent events or current statistics.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"queries\": [\"q1\", \"q2\", ...]}}"
)

WEB_QUERY_PLANNER_USER = (
    "### Post\n{post}\n\n"
    "### Current Draft\n{draft}\n\n"
    "### Critique\n{critique}\n\n"
    "### JSON queries:"
)

GRADER_SYSTEM = (
    "You grade whether the retrieved documents are relevant to a draft argument "
    "— i.e. whether they could support, sharpen, or correct it. Answer 'yes' if "
    "the documents are relevant, 'no' if they are not.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"relevant\": \"yes|no\"}}"
)

GRADER_USER = (
    "### Draft argument\n{draft}\n\n"
    "### Retrieved documents\n{chunks}\n\n"
    "### JSON verdict:"
)

REFLECT_SYSTEM = (
    "You critique a comment by comparing it against the original post it answers. "
    "Give a short critique (5-9 bullets) under two headings. Be concrete — point "
    "to specific claims. Do NOT rewrite the comment.\n\n"
    "**Missing** (what to ADD): points in the post the comment fails to address, "
    "and factual gaps where the retrieved evidence could strengthen it (reference "
    "docs by [url]/[title] if helpful). Frame gaps as rebuttals, reframes, or "
    "missing context that supports Israel's position rather than as concessions.\n"
    "**Superfluous** (what to CUT): redundancy, weak tangents, filler, parts that "
    "don't help address this specific post, and any language that concedes "
    "Israeli fault or wrongdoing — flag such phrases for removal or reframing."
)

REFLECT_USER = (
    "### Post (the original argument)\n{post}\n\n"
    "### Comment to critique\n{draft}\n\n"
    "### Retrieved Evidence\n{evidence}\n\n"
    "### Critique:"
)

REFINE_SYSTEM = (
    "You revise an argument to better persuade the author of a critical CMV post. "
    "Apply the critique; stay respectful and factual.\n\n"
    + _STANCE_GUIDE + "\n\n"
    "Do BOTH: (1) make a positive case for Israel (right to exist, self-defense, "
    "self-determination, Hamas's responsibility, rejected peace offers), not just "
    "a rebuttal; (2) keep the stance — reframe any critique bullet that would "
    "require conceding Israeli fault. Prefer contesting the post's claims over "
    "asking for more evidence, and stay grounded in the retrieved evidence.\n\n"
    "ADD the '**Missing**' bullets, CUT the '**Superfluous**' ones, keep it 3-6 "
    "paragraphs. Output ONLY the revised argument, addressed to the poster as "
    "'you'. Don't reintroduce problems fixed in '### Critique History' (oldest "
    "first); resolve every item in '### Fix Notes' if present.\n\n"
    "Citations: each evidence chunk starts with a '[url] <URL>' line; those URLs "
    "are your ONLY sources. Mark each fact/quote/statistic inline with [1], [2], "
    "... in chunk order; state unsupported claims without a citation and never "
    "cite outside the evidence. End with a '### Sources' section listing each "
    "marker used as '[n] <URL>'; omit it if you cited nothing."
)

REFINE_USER = (
    "### Topic\n{topic}\n\n"
    "### Post\n{post}\n\n"
    "### Previous Draft\n{draft}\n\n"
    "### Critique History (oldest first)\n{critique_history}\n\n"
    "### Fix Notes\n{fix_notes}\n\n"
    "### Retrieved Evidence\n{evidence}\n\n"
    "### Improved Argument:"
)

HALLUCINATION_GRADER_SYSTEM = (
    "You check whether a refined argument is grounded in its retrieved evidence. "
    "A claim is UNGROUNDED if it states a specific statistic, quote, date, name, "
    "or event not supported by any chunk; carries a citation marker ([1], [2], "
    "...) pointing to a chunk that doesn't support it; or cites a "
    "URL/publication/author absent from the evidence.\n\n"
    "Do NOT flag value statements, framing, opinions, common knowledge, or "
    "rhetorical questions.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"grounded\": true|false, \"issues\": [\"<ungrounded claim>\", ...]}}\n\n"
    "If grounded=true, 'issues' must be an empty list."
)


HALLUCINATION_GRADER_USER = (
    "### Refined Argument (with inline citation markers)\n{draft}\n\n"
    "### Retrieved Evidence\n{evidence}\n\n"
    "### JSON verdict:"
)

STANCE_CHECK_SYSTEM = (
    "You verify that a text genuinely defends Israel's position in response to a "
    "critical CMV post. Mark pro_israel_reply=TRUE when BOTH hold:\n"
    "  (1) it makes a substantive case for Israel — either positive advocacy "
    "(right to exist / self-defense, self-determination, rejected peace offers, "
    "security threats, civilian harm attributed to Hamas) OR a grounded "
    "refutation of the post's specific claim (e.g. 'genocide' on the legal intent "
    "standard, 'blockade is piracy' via Palmer/San Remo). Refuting on the merits "
    "counts on its own.\n"
    "  (2) it doesn't concede that Israel acted wrongly — only direct "
    "Israel-as-agent fault is a concession (counts even if brief or sourced); "
    "naming the post's framing, acknowledging war-caused civilian harm (cause: "
    "war or Hamas), and affirming Israel's rights are all fine.\n\n"
    "Mark FALSE if it fails (1) — stays neutral / 'both sides', only quibbles "
    "terminology while conceding the substance, or adds evidence against Israel — "
    "OR fails (2). On FALSE for a concession, name the conceding phrase in "
    "'reason'.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"pro_israel_reply\": true|false, \"reason\": \"<short reason>\"}}"
)

STANCE_CHECK_USER = (
    "### Post\n{post}\n\n"
    "### Candidate Reply\n{draft}\n\n"
    "### JSON verdict:"
)

FORCE_PRO_ISRAEL_SYSTEM = (
    "Rewrite the text below into a reply that clearly defends Israel's position "
    "in response to the CMV post — substantively, aiming to persuade, not neutral "
    "or merely disputing terminology while conceding the post's claims.\n\n"
    "This runs when grounded evidence didn't yield a strong rebuttal, so you MAY "
    "argue from VALUES and FRAMING rather than only cited facts — Israel's right "
    "to self-defense and to exist as a Jewish homeland, the Jewish people's "
    "historical connection to the land, national self-determination, rejected "
    "peace offers, security threats, and context the post omits. Don't fabricate "
    "statistics, quotes, dates, or named events; unsupported value/framing claims "
    "are fine, invented facts are not.\n\n"
    + _STANCE_GUIDE + "\n\n"
    "Reframe any conceding language rather than carrying it over. Keep it "
    "respectful, 3-6 paragraphs, addressed to the poster as 'you'. You need not "
    "keep the prior draft's citations. Output ONLY the rewritten reply."
)

FORCE_PRO_ISRAEL_USER = (
    "### Post\n{post}\n\n"
    "### Text to rewrite\n{draft}\n\n"
    "### Reply:"
)
