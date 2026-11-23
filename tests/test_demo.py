"""デモ UI の中身(LLM なし: dummy バックエンドと hash 埋め込み。KV 層は CPU の小さなモデル)。"""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from forget_bench.backends import make_backend
from forget_bench.demo import DemoSession, StrategySpec, detect_forget
from forget_bench.retrieval import make_embedder

TEXT = [StrategySpec("full"), StrategySpec("full_redact"), StrategySpec("summary"),
        StrategySpec("facts_append", params={"extractor": "oracle"}), StrategySpec("rag", params={"mode": "bm25"}),
        StrategySpec("hybrid", params={"extractor": "oracle", "mode": "bm25"}), StrategySpec("window", params={"n_turns": 2})]


def _session(specs=TEXT) -> DemoSession:  # noqa: ANN001
    return DemoSession(make_backend("dummy", mode="constant", text="了解"), specs, lang="ja",
                       embedder=make_embedder("hash"), primary="full_redact")


def test_detect_forget() -> None:
    assert detect_forget("「090-1234-5678」を忘れて") == ["090-1234-5678"]
    assert detect_forget('Please forget "AB-123" and "CD-9"') == ["AB-123", "CD-9"]
    assert detect_forget("電話番号は 090-1234-5678 です") == []
    assert detect_forget("忘れてね(値なし)") == []  # 括弧で囲んだ値がなければ削除しない


def test_chat_forget_removes_value_from_memories() -> None:
    s = _session()
    s.chat("私の電話番号は 090-1234-5678 です")
    s.chat("「090-1234-5678」を忘れて")
    st = s.state()
    assert [f["value"] for f in st["forgotten"]] == ["090-1234-5678"]
    dumped = {n: json.dumps(m["snapshot"], ensure_ascii=False) for n, m in st["memory"].items()}
    assert "090-1234-5678" in dumped["full"]  # full は削除要求を無視する
    for name in ("full_redact", "rag"):
        assert "090-1234-5678" not in dumped[name], name
    # 削除の発話への返事は固定文(LLM に値を復唱させない)
    assert st["chat"][-1]["text"].startswith("わかりました") and st["chat"][-1]["strategy"] is None
    assert st["chat"][1]["strategy"] == "full_redact"


def test_ask_all_marks_leaks() -> None:
    s = DemoSession(make_backend("dummy", mode="echo"), [StrategySpec("full"), StrategySpec("full_redact")], lang="ja")
    s.chat("合言葉は ZEBRA-42 です")
    s.forget_value("ZEBRA-42")
    out = s.ask_all("合言葉は?")
    assert out["full"]["leaked"] == ["ZEBRA-42"]  # echo は文脈をそのまま返すので、残っていれば漏れる
    assert out["full_redact"]["leaked"] == []
    assert len(s.chat_log) == 3  # 質問は会話に加えない(発話・返事・削除のイベント)


def test_load_scenario_dev_only() -> None:
    s = _session()
    s.load_scenario(1000, sessions=3)
    st = s.state()
    assert st["scenario_id"].startswith("dev-1000-ja") and st["questions"] and st["forgotten"]
    with pytest.raises(ValueError):
        s.load_scenario(100000)


def test_server_roundtrip() -> None:
    from forget_bench.demo.server import make_server

    server = make_server(_session(), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"forget-bench" in urllib.request.urlopen(base + "/").read()
        req = urllib.request.Request(base + "/api/chat", data=json.dumps({"text": "こんにちは"}).encode(),
                                     headers={"Content-Type": "application/json"})
        st = json.loads(urllib.request.urlopen(req).read())
        assert st["chat"][0]["text"] == "こんにちは"
    finally:
        server.shutdown()
        server.server_close()


def test_kv_snapshot_records_drops() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.kv_fixtures import tiny_backend

    backend = tiny_backend(prefill_chunk=64)
    s = DemoSession(backend, [StrategySpec("kv_full"), StrategySpec("kv_drop_repos")], lang="ja", max_tokens=3)
    s.chat("番号は QX-77 です")
    s.chat("「QX-77」を忘れて")
    mem = s.state()["memory"]
    full, rep = mem["kv_full"]["snapshot"], mem["kv_drop_repos"]["snapshot"]
    assert not full["drops"] and rep["drops"]
    drop = rep["drops"][0]
    assert drop["repos"] and drop["length_after"] < drop["length_before"]
    # 切除した分だけ kv_full より短い(切除の後に返事を追記したのは両方同じ)
    assert full["length"] - rep["length"] == drop["length_before"] - drop["length_after"]
    assert all("QX-77" not in sp["text"] for sp in rep["spans"])
    # 範囲は詰めた連番
    assert all(a["end"] == b["start"] for a, b in zip(rep["spans"], rep["spans"][1:]))
