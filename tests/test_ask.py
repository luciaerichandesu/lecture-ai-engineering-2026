from types import SimpleNamespace

import app.ask as ask_module


def _mock_client(monkeypatch, response):
    models = SimpleNamespace(generate_content=lambda **_: response)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: SimpleNamespace(models=models))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12, candidates_token_count=34, total_token_count=46
        ),
    )
    _mock_client(monkeypatch, response)
    timestamps = iter([1.0, 2.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(timestamps))

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out == (
        "[usage] input_tokens=12, output_tokens=34, total_tokens=46\n"
        "[response_time] 1.25s\n"
    )


def test_ask_reports_unavailable_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    _mock_client(monkeypatch, response)
    timestamps = iter([1.0, 2.0])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(timestamps))

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out == (
        "[usage] input_tokens=不明, output_tokens=不明, total_tokens=不明\n"
        "[response_time] 1.00s\n"
    )


def test_ask_calculates_total_tokens_when_api_total_is_unavailable(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
    )
    _mock_client(monkeypatch, response)
    timestamps = iter([1.0, 1.5])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(timestamps))

    ask_module.ask("question")

    assert "total_tokens=46" in capsys.readouterr().out


def test_ask_includes_conversation_history(monkeypatch):
    received = {}
    response = SimpleNamespace(text="second answer", usage_metadata=None)

    def generate_content(**kwargs):
        received.update(kwargs)
        return response

    models = SimpleNamespace(generate_content=generate_content)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: SimpleNamespace(models=models))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    history = [
        ask_module.types.Content(
            role="user", parts=[ask_module.types.Part.from_text(text="first question")]
        ),
        ask_module.types.Content(
            role="model", parts=[ask_module.types.Part.from_text(text="first answer")]
        ),
    ]

    ask_module.ask("second question", history)

    contents = received["contents"]
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert [content.parts[0].text for content in contents] == [
        "first question",
        "first answer",
        "second question",
    ]


def test_main_keeps_history_between_questions_and_exits(monkeypatch, capsys):
    received = []
    responses = iter(["first answer", "second answer"])

    def generate_content(**kwargs):
        received.append(kwargs["contents"])
        return SimpleNamespace(
            text=next(responses),
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
        )

    models = SimpleNamespace(generate_content=generate_content)
    monkeypatch.setattr(ask_module.genai, "Client", lambda **_: SimpleNamespace(models=models))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    prompts = iter(["second question", "終了"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    ask_module.main()

    assert [[content.role for content in turn] for turn in received] == [
        ["user"],
        ["user", "model", "user"],
    ]
    assert "first answer" in capsys.readouterr().out
