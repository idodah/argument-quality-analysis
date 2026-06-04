"""Gradio app: paste a CMV topic + anti-Israel post, get a pro-Israel rebuttal.

Run:
    uv run python -m webapp.app

Reuses agents.generate.generate_pro_israel_response, so the web UI, the CLI, and
the harvester all run the SAME agentic graph — no duplicated logic. No Reddit
access is needed: the user pastes the post text directly.
"""

from __future__ import annotations

import traceback

import gradio as gr
from dotenv import load_dotenv

from agents.generate import generate_pro_israel_response

load_dotenv()

_EXAMPLE_TOPIC = "CMV: Israel is committing a genocide in Gaza"
_EXAMPLE_POST = (
    "Using the UN definition of genocide, I believe Israel's conduct in Gaza meets the "
    "threshold. The scale of civilian deaths, the destruction of infrastructure, and "
    "statements by officials all point to intent to destroy the population. Change my view."
)


def _format_sources(urls: list[str]) -> str:
    """Markdown for the human-in-the-loop sources panel (kept out of the text)."""
    if not urls:
        return "_No sources cited (argued from reasoning, not retrieved facts)._"
    return "\n".join(f"{i}. {u}" for i, u in enumerate(urls, 1))


def respond(topic: str, post: str):
    """Run the graph and return (clean_argument, status_markdown, sources_markdown)."""
    topic = (topic or "").strip()
    post = (post or "").strip()
    if not post:
        return "", "Please paste the original post body.", ""
    if not topic:
        topic = "CMV"  # topic is used as a hint; a sensible default is fine

    try:
        result = generate_pro_israel_response(title=topic, body=post)
    except Exception as e:  # surface failures in the UI rather than a blank box
        # Show a clean one-liner in the UI, but log the full traceback to the
        # server console so a real bug isn't reduced to a single line.
        traceback.print_exc()
        return "", f"Generation failed: {type(e).__name__}: {e}", ""

    flags = []
    if not result["grounded"]:
        flags.append("not fully grounded in evidence")
    if not result["pro_israel_reply"]:
        flags.append("not a clearly pro-Israel reply")
    status = (
        "Warning — " + "; ".join(flags) + ". Review before using."
        if flags
        else "Grounded and pro-Israel."
    )
    # `generation` is already cleaned of inline markers + the Sources footer;
    # `sources` is surfaced separately for manual verification.
    return result["generation"], status, _format_sources(result.get("sources", []))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Pro-Israel Argument Generator") as demo:
        gr.Markdown(
            "# Pro-Israel Argument Generator\n"
            "Paste a r/ChangeMyView **topic** and the **anti-Israel post**, and the agentic "
            "workflow (Adaptive/Self/Reflective RAG + Reflexion, with a Qwen ranker) drafts a "
            "respectful, sourced rebuttal.\n\n"
            "_A full run loads a local model and makes several LLM + web-search calls, so it can "
            "take from tens of seconds to a couple of minutes._"
        )
        with gr.Row():
            with gr.Column():
                topic = gr.Textbox(label="Topic / CMV title", value=_EXAMPLE_TOPIC, lines=1)
                post = gr.Textbox(label="Anti-Israel post body", value=_EXAMPLE_POST, lines=12)
                go = gr.Button("Generate response", variant="primary")
            with gr.Column():
                status = gr.Markdown(label="Status")
                argument = gr.Textbox(label="Generated argument (no citations)", lines=20)
                with gr.Accordion("Sources (for your review — not shown in the argument)", open=False):
                    sources = gr.Markdown()

        go.click(fn=respond, inputs=[topic, post], outputs=[argument, status, sources])
    return demo


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
