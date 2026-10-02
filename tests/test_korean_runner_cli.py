from scripts.run_configuration import build_parser


def test_runner_accepts_explicit_korean_prompt_version() -> None:
    args = build_parser().parse_args(
        [
            "C01",
            "--base-url",
            "http://127.0.0.1:8201/v1",
            "--prompt-version",
            "vqa-generation-ko-v1",
        ]
    )

    assert args.prompt_version == "vqa-generation-ko-v1"
