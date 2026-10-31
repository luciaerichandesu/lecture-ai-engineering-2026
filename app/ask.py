"""第1回: 会話履歴を保ちながら素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問の後、続けて質問できる

「終了」、"exit"、"quit" を入力すると終了する。
ツールは持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
EXIT_COMMANDS = {"終了", "exit", "quit"}


def ask(question: str, history: list[types.Content] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    contents = [
        *(history or []),
        types.Content(role="user", parts=[types.Part.from_text(text=question)]),
    ]
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=contents)
    elapsed = time.perf_counter() - started_at

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    input_tokens = input_tokens if input_tokens is not None else "不明"
    output_tokens = output_tokens if output_tokens is not None else "不明"
    total_tokens = total_tokens if total_tokens is not None else "不明"
    print(
        f"[usage] input_tokens={input_tokens}, output_tokens={output_tokens}, "
        f"total_tokens={total_tokens}"
    )
    print(f"[response_time] {elapsed:.2f}s")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保ちながら素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    history: list[types.Content] = []
    question = args.question
    try:
        while question.strip().lower() not in EXIT_COMMANDS:
            print(f"[質問] {question}\n")
            answer = ask(question, history)
            print(answer)
            history.extend(
                [
                    types.Content(role="user", parts=[types.Part.from_text(text=question)]),
                    types.Content(role="model", parts=[types.Part.from_text(text=answer)]),
                ]
            )
            question = input("質問（終了するには「終了」）> ")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
