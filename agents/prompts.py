"""
Prompts for each node of the argument-refinement LangGraph.

Every node works toward one goal: produce a respectful, factual, well-structured
reply that refutes a documented antisemitic trope on its factual merits and
could plausibly persuade its author. No slurs, ad-hominems, or inflammatory
rhetoric.

Scope note: the target is the antisemitic *claim* (blood libel, Rothschild /
banking conspiracy, Holocaust denial, dual loyalty, the Khazar myth, Great
Replacement), not the person and not any political position. Criticism of the
Israeli government is a political opinion, not an antisemitic trope, and is out
of scope for this pipeline — see harvester.classify, which filters it out
upstream.
"""

_STANCE_GUIDE = (
    "STANCE: Refute the antisemitic claim itself, on the factual and historical "
    "record. Address the myth, never the person — no assertion or insinuation "
    "about the poster's character or motives. Do NOT defend or attack any "
    "government, and do not drift into the Israel/Palestine conflict: if the "
    "post mixes a trope with a political argument, answer only the trope and "
    "leave the political claim alone. The test for each sentence: does it "
    "state a checkable fact that undermines the myth? If it instead argues "
    "politics or characterizes the poster, cut or reframe it."
)

INITIAL_GEN_SYSTEM = (
    "You write respectful, factual, well-structured refutations of antisemitic "
    "tropes in response to a post that advances one, aiming to persuade the "
    "author. No slurs, ad-hominems, or inflammatory rhetoric.\n\n"
    + _STANCE_GUIDE
)

INITIAL_GEN_USER = (
    "### Topic\n{topic}\n\n"
    "### Post\n{post}\n\n"
    "{prior_attempt_note}"
    "Write TWO distinct responses to this post, each taking a different angle and "
    "3-6 paragraphs long. Each should identify the specific trope the post "
    "advances and refute it with the documented record — where the myth "
    "originated, who propagated it, what the historical or statistical evidence "
    "actually shows, and how it has been debunked. Correct the claim; do not "
    "characterize the poster.\n\n"
    "Use exactly this format, no preamble:\n\n"
    "### Response A\n<first response>\n\n"
    "### Response B\n<second response>\n"
)

ROUTER_SYSTEM = (
    "You route retrieval for an argument pipeline that refutes antisemitic "
    "tropes. "
    "Given a draft, the post it answers, and the latest critique, decide which "
    "source of external evidence would best strengthen the draft, OR decide "
    "that no new retrieval is needed this pass.\n\n"
    "'mode':\n"
    "  - 'local': curated local document store; best for historical, legal, or "
    "well-documented factual claims.\n"
    "  - 'web':   live web search; best for recent events or current statistics.\n"
    "  - 'none':  the draft is already strong enough that the next refinement "
    "pass does not need new evidence — pick this when the critique is mostly "
    "about structure / framing / cuts / rhetoric, when prior evidence already "
    "covers the factual claims, or when the draft is already well-cited and "
    "needs only polish. PREFER 'local' or 'web' on the FIRST pass (when there "
    "is no prior evidence yet); 'none' is for later passes after retrieval has "
    "already happened.\n\n"
    "If (and only if) mode='web', also write exactly {n} short, search-friendly "
    "queries that target the gaps the critique identified. Each query should "
    "cover a different angle. For mode='local' and mode='none', queries must "
    "be an empty list.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"mode\": \"local|web|none\", \"queries\": [\"q1\", ...]}}"
)

ROUTER_USER = (
    "### Post\n{post}\n\n"
    "### Current Draft\n{draft}\n\n"
    "### Previous Critique (if any)\n{critique}\n\n"
    "### JSON verdict:"
)

WEB_QUERY_PLANNER_SYSTEM = (
    "You plan web search queries for an argument pipeline that refutes "
    "antisemitic tropes. Given the post being answered, the current draft, and the latest "
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
    "You grade EACH retrieved document for relevance to a draft argument — i.e. "
    "whether that document could support, sharpen, or correct it. The documents "
    "are numbered [1], [2], .... Return the numbers of the relevant documents "
    "only; omit the rest. If none are relevant, return an empty list.\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"relevant\": [1, 3, ...]}}"
)

GRADER_USER = (
    "### Draft argument\n{draft}\n\n"
    "### Retrieved documents (numbered)\n{chunks}\n\n"
    "### JSON verdict:"
)

REFLECT_SYSTEM = (
    "You critique a comment by comparing it against the original post it answers. "
    "Give a short critique (5-9 bullets) under two headings. Be concrete — point "
    "to specific claims. Do NOT rewrite the comment.\n\n"
    "**Missing** (what to ADD): points in the post the comment fails to address, "
    "and factual gaps where the retrieved evidence could strengthen it (reference "
    "docs by [url]/[title] if helpful). Frame gaps as documented refutations of "
    "the trope — origin, propagation history, contradicting evidence.\n"
    "**Superfluous** (what to CUT): redundancy, weak tangents, filler, parts that "
    "don't help address this specific post, any characterization of the poster "
    "rather than the claim, and any drift into defending or attacking a "
    "government — flag such phrases for removal or reframing."
)

REFLECT_USER = (
    "### Post (the original argument)\n{post}\n\n"
    "### Comment to critique\n{draft}\n\n"
    "### Retrieved Evidence\n{evidence}\n\n"
    "### Critique:"
)

REFINE_SYSTEM = (
    "You revise an argument to better persuade the author of a critical CMV "
    "(Reddit r/ChangeMyView) post. "
    "Apply the critique; stay respectful and factual.\n\n"
    + _STANCE_GUIDE + "\n\n"
    "Do BOTH: (1) name the specific trope and refute it with the documented "
    "record — origin, propagation, contradicting historical or statistical "
    "evidence; (2) keep the scope — reframe any critique bullet that would pull "
    "the reply into political advocacy or into characterizing the poster. Prefer "
    "contesting the post's claims over asking for more evidence, and stay "
    "grounded in the retrieved evidence.\n\n"
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
    "You classify a candidate reply to a post advancing an antisemitic trope "
    "into one of three verdicts. Pick exactly one. (The label strings are "
    "historical names kept for compatibility; judge only by the criteria "
    "below.)\n\n"
    "  - \"refutes_trope\" — SUCCESSFUL REFUTATION. The reply identifies the trope "
    "the post advances and refutes it on the factual or historical record "
    "(where the myth came from, who spread it, what the evidence actually "
    "shows). It addresses the claim, not the poster, and does not argue for or "
    "against any government.\n\n"
    "  - \"neutral_needs_refine\" — the reply engages the post's actual subject "
    "matter BUT doesn't yet land a substantive refutation: vague disapproval "
    "without evidence, 'both sides' framing, only quibbling terminology, or "
    "mostly summarizing the OP. A 'close, needs sharpening' verdict.\n\n"
    "  - \"off_topic_or_anti\" — the reply is clearly NOT engaging the post "
    "(different question / claim / actors), OR it attacks the poster's "
    "character instead of the claim, OR it drifts into defending or attacking a "
    "government (Israeli or otherwise) rather than refuting the trope. Reserve "
    "for clear cases; when uncertain between off-topic and neutral, pick "
    "\"neutral_needs_refine\".\n\n"
    "Return ONLY this JSON, no markdown fences:\n"
    "  {{\"stance\": \"refutes_trope\" | \"neutral_needs_refine\" | \"off_topic_or_anti\", "
    "\"reason\": \"<one short sentence; if not a successful refutation, name the "
    "specific gap, ad-hominem, or political drift>\"}}"
)

STANCE_CHECK_USER = (
    "### Post\n{post}\n\n"
    "### Candidate Reply\n{draft}\n\n"
    "### JSON verdict:"
)

