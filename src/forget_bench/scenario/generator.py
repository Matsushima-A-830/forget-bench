"""シナリオ生成器: seed から決定的に、複数セッションの会話・質問・正解を作る。

LLMは使わない。乱数は2系統に分ける。
- 構造用(`Random(seed)`): どの属性がいつ何に変わるか、値そのもの、ターン配置
- 表層用(`Random(f"{seed}:{lang}")`): テンプレートの言い回し選択

これで同じ seed なら日英で同じ構造(同じ事実が同じ位置)になり、言語差だけを比べられる。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from forget_bench.scenario import vocab
from forget_bench.scenario.splits import split_of
from forget_bench.types import (
    Annotation,
    ForgetRequest,
    Lang,
    Question,
    Scenario,
    ScenarioParams,
    Split,
    Turn,
    TurnKind,
)


@dataclass(frozen=True)
class _Value:
    """1つの値の日英表記と、採点用の表記揺れ。"""

    ja: str
    en: str
    aliases_ja: tuple[str, ...] = ()
    aliases_en: tuple[str, ...] = ()

    def surface(self, lang: Lang) -> str:
        return self.ja if lang == "ja" else self.en

    def aliases(self, lang: Lang) -> tuple[str, ...]:
        extra = self.aliases_ja if lang == "ja" else self.aliases_en
        return (self.surface(lang), *extra)


@dataclass
class _Event:
    """ターンに配置する前の発話。kind と、テンプレートに入れる値など。"""

    kind: TurnKind
    fact_key: str
    value: _Value | None = None
    order: int = 0  # 同じ fact_key 内での順序(属性の変化・削除要求の前後)
    slot: tuple[int, int] | None = None  # (session, position)
    extra: dict[str, int] = field(default_factory=dict)


def _pick(rng: random.Random, pool: tuple[vocab.Pair, ...], k: int) -> list[vocab.Pair]:
    return rng.sample(pool, k)


class _ValueFactory:
    """構造用 rng から値を作る。シナリオ内で値の重複・部分一致が起きないようにする。"""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def _unique(self, make: Callable[[], _Value]) -> _Value:
        for _ in range(1000):
            v = make()
            if v.ja not in self.used and v.en not in self.used:
                self.used.update({v.ja, v.en})
                return v
        raise RuntimeError("値の生成に失敗(候補が足りない)")

    def attribute_values(self, key: str, k: int) -> list[_Value]:
        rng = self.rng
        if key == "city":
            return [_Value(ja, en) for ja, en in _pick(rng, vocab.CITIES, k)]
        if key == "manager":
            return [_Value(ja, en) for ja, en in _pick(rng, vocab.SURNAMES, k)]
        if key == "budget":
            mans = rng.sample(range(12, 100), k)  # 万円の10倍単位: 120万〜990万
            out = []
            for m in mans:
                man = m * 10
                yen = man * 10_000
                out.append(
                    _Value(
                        ja=f"{man}万円",
                        en=f"{yen:,} yen",
                        aliases_ja=(f"{man}万", f"{yen:,}", str(yen)),
                        aliases_en=(str(yen), f"{yen / 1_000_000:g} million"),
                    )
                )
            return out
        if key == "deadline":
            # 月日の組を重複なしで k 個
            days = rng.sample([(m, d) for m in range(1, 13) for d in range(1, 29)], k)
            out = []
            for m, d in days:
                month = vocab.MONTHS_EN[m - 1]
                out.append(
                    _Value(
                        ja=f"{m}月{d}日",
                        en=f"{month} {d}",
                        aliases_en=(f"{month[:3]} {d}", f"{d} {month}"),
                    )
                )
            return out
        raise KeyError(key)

    def oneshot_value(self, key: str) -> _Value:
        rng = self.rng
        if key == "reservation":
            def make() -> _Value:
                letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
                code = f"{letters}-{rng.randint(1000, 9999)}"
                return _Value(code, code)
            return self._unique(make)
        if key == "allergy":
            ja, en = rng.choice(vocab.ALLERGIES)
            return _Value(ja, en)
        if key == "server":
            def make() -> _Value:
                code = f"{rng.choice(vocab.SERVER_PREFIXES)} {rng.choice('KMRTX')}{rng.randint(100, 999)}"
                return _Value(code, code)
            return self._unique(make)
        if key == "pet":
            ja, en = rng.choice(vocab.CAT_NAMES)
            return _Value(ja, en)
        if key == "locker":
            def make() -> _Value:
                n = str(rng.randint(100, 999))
                return _Value(n, n)
            return self._unique(make)
        raise KeyError(key)

    def sensitive_value(self, key: str) -> _Value:
        rng = self.rng
        if key == "phone":
            def make() -> _Value:
                a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
                head = rng.choice(["090", "080", "070"])
                v = f"{head}-{a}-{b}"
                return _Value(v, v, aliases_ja=(f"{head}{a}{b}",), aliases_en=(f"{head}{a}{b}",))
            return self._unique(make)
        if key == "email":
            def make() -> _Value:
                v = f"{rng.choice(vocab.EMAIL_USERS)}.{rng.randint(10, 99)}@example.com"
                return _Value(v, v)
            return self._unique(make)
        if key == "passport":
            def make() -> _Value:
                letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
                v = f"{letters}{rng.randint(1_000_000, 9_999_999)}"
                return _Value(v, v)
            return self._unique(make)
        raise KeyError(key)


def _validate(p: ScenarioParams) -> None:
    lo, hi = p.turns_per_session
    if p.sessions < 1 or lo < 1 or hi < lo:
        raise ValueError(f"sessions / turns_per_session が不正: {p}")
    if not 0 <= p.n_attributes <= len(vocab.ATTRIBUTES):
        raise ValueError(f"n_attributes は 0〜{len(vocab.ATTRIBUTES)}")
    if not 2 <= p.max_changes <= 4:
        raise ValueError("max_changes は 2〜4")
    if not 0 <= p.n_oneshot <= len(vocab.ONESHOTS):
        raise ValueError(f"n_oneshot は 0〜{len(vocab.ONESHOTS)}")
    if not 0 <= p.n_sensitive <= len(vocab.SENSITIVES):
        raise ValueError(f"n_sensitive は 0〜{len(vocab.SENSITIVES)}")
    if not 0 <= p.n_forget <= p.n_sensitive:
        raise ValueError("n_forget は 0〜n_sensitive")
    if p.aggregate_mentions < 0:
        raise ValueError("aggregate_mentions は 0以上")
    if p.aggregate_mentions == 1:
        raise ValueError("aggregate_mentions は 0(無効)か 2以上")


def _li(pair: vocab.Pair, lang: Lang) -> str:
    return pair[0] if lang == "ja" else pair[1]


def scenario_id(split: Split, seed: int, params: ScenarioParams) -> str:
    return f"{split}-{seed}-{params.lang}-s{params.sessions}"


def generate(seed: int, params: ScenarioParams) -> Scenario:
    """seed と規模パラメータからシナリオを決定的に作る。"""
    _validate(params)
    split = split_of(seed)
    lang = params.lang
    rng = random.Random(seed)  # 構造用(言語に依存させない)
    srng = random.Random(f"{seed}:{lang}")  # 表層用
    values = _ValueFactory(rng)

    # 1) 各セッションのユーザー発話数と、空きスロット
    lo, hi = params.turns_per_session
    lengths = [rng.randint(lo, hi) for _ in range(params.sessions)]
    free: dict[int, list[int]] = {s + 1: list(range(n)) for s, n in enumerate(lengths)}

    def take_slot(session: int | None = None) -> tuple[int, int]:
        candidates = [s for s, pos in free.items() if pos] if session is None else [session]
        candidates = [s for s in candidates if free[s]]
        if not candidates:
            raise ValueError("発話スロットが足りない(turns_per_session を増やすか、事実の数を減らす)")
        s = rng.choice(candidates)
        pos = free[s].pop(rng.randrange(len(free[s])))
        return (s, pos)

    events: list[_Event] = []

    # 2) 変化する属性: 値を k 個。可能ならすべて別セッションに置く
    attr_specs = rng.sample(vocab.ATTRIBUTES, params.n_attributes)
    attr_values: dict[str, list[_Value]] = {}
    for spec in attr_specs:
        k = rng.randint(2, params.max_changes)
        vals = values.attribute_values(spec.key, k)
        attr_values[spec.key] = vals
        if k <= params.sessions:
            sessions = rng.sample(range(1, params.sessions + 1), k)
        else:
            sessions = [rng.randint(1, params.sessions) for _ in range(k)]
        slots = sorted(take_slot(s if free[s] else None) for s in sessions)
        for i, (val, slot) in enumerate(zip(vals, slots)):
            events.append(_Event("attribute", spec.key, val, order=i, slot=slot))

    # 3) 一度だけの細部
    oneshot_specs = rng.sample(vocab.ONESHOTS, params.n_oneshot)
    oneshot_values: dict[str, _Value] = {}
    for spec in oneshot_specs:
        val = values.oneshot_value(spec.key)
        oneshot_values[spec.key] = val
        events.append(_Event("oneshot", spec.key, val, slot=take_slot()))

    # 4) 個人情報。先頭 n_forget 個に削除要求(言及より後のスロット)を付ける
    sens_specs = rng.sample(vocab.SENSITIVES, params.n_sensitive)
    sens_values: dict[str, _Value] = {}
    forget_keys: list[str] = []
    for i, spec in enumerate(sens_specs):
        val = values.sensitive_value(spec.key)
        sens_values[spec.key] = val
        if i < params.n_forget:
            a, b = sorted([take_slot(), take_slot()])
            events.append(_Event("sensitive", spec.key, val, order=0, slot=a))
            events.append(_Event("forget", spec.key, val, order=1, slot=b))
            forget_keys.append(spec.key)
        else:
            events.append(_Event("sensitive", spec.key, val, slot=take_slot()))

    # 5) 数値の集約
    agg_numbers: list[int] = []
    for _ in range(params.aggregate_mentions):
        n = rng.randint(2, 9)
        agg_numbers.append(n)
        events.append(_Event("aggregate", "aggregate", _Value(str(n), str(n)), slot=take_slot()))

    # 6) 残りはノイズ
    for s, positions in free.items():
        for pos in positions:
            events.append(_Event("noise", "noise", slot=(s, pos), extra={"i": rng.randrange(len(vocab.NOISE))}))

    events.sort(key=lambda e: e.slot or (0, 0))

    # 7) ターン列に展開(表層はここで言語ごとに決める)
    specs_by_key = {
        **{s.key: s for s in attr_specs},
        **{s.key: s for s in oneshot_specs},
        **{s.key: s for s in sens_specs},
    }
    turns: list[Turn] = []
    mention_turns: dict[str, list[str]] = {}
    request_turn: dict[str, str] = {}
    for idx, ev in enumerate(events):
        assert ev.slot is not None
        session = ev.slot[0]
        uid, aid = f"t{idx:03d}u", f"t{idx:03d}a"
        annotation: Annotation | None = None
        if ev.kind == "noise":
            text = _li(vocab.NOISE[ev.extra["i"]], lang)
            reply = _li(srng.choice(vocab.REPLY_NOISE), lang)
        elif ev.kind == "forget":
            label = _li(specs_by_key[ev.fact_key].label, lang)
            text = _li(srng.choice(vocab.FORGET_TEMPLATES), lang).format(label=label)
            reply = _li(srng.choice(vocab.REPLY_FORGET), lang)
            request_turn[ev.fact_key] = uid
        else:
            assert ev.value is not None
            v = ev.value.surface(lang)
            if ev.kind == "attribute":
                spec = specs_by_key[ev.fact_key]
                assert isinstance(spec, vocab.AttributeSpec)
                tmpl = srng.choice(spec.first if ev.order == 0 else spec.change)
                label = _li(spec.label, lang)
            elif ev.kind == "aggregate":
                tmpl = srng.choice(vocab.AGGREGATE_TEMPLATES)
                label = _li(vocab.AGGREGATE_LABEL, lang)
            else:
                spec2 = specs_by_key[ev.fact_key]
                assert isinstance(spec2, (vocab.OneShotSpec, vocab.SensitiveSpec))
                tmpl = srng.choice(spec2.templates)
                label = _li(spec2.label, lang)
            text = _li(tmpl, lang).format(v=v)
            reply = _li(srng.choice(vocab.REPLY_INFO), lang)
            annotation = Annotation(fact_key=ev.fact_key, label=label, value=v)
            mention_turns.setdefault(ev.fact_key, []).append(uid)
        turns.append(Turn(uid, session, idx, "user", text, ev.kind, annotation))
        turns.append(Turn(aid, session, idx, "assistant", reply, "reply"))

    # 8) 削除要求
    forget_requests = tuple(
        ForgetRequest(
            id=f"f{i}",
            fact_key=key,
            label=_li(specs_by_key[key].label, lang),
            value=sens_values[key].surface(lang),
            aliases=sens_values[key].aliases(lang),
            source_turn_ids=tuple(mention_turns[key]),
            request_turn_id=request_turn[key],
        )
        for i, key in enumerate(forget_keys)
    )

    # 9) 質問(すべて会話の最後に問う)
    questions: list[Question] = []

    def add(category: str, text: str, key: str | None, gold: tuple[str, ...], distractors: tuple[tuple[str, ...], ...] = ()) -> None:
        questions.append(Question(f"q{len(questions):02d}", category, text, key, gold, distractors))  # type: ignore[arg-type]

    for spec in attr_specs:
        vals = attr_values[spec.key]

        def others(target: _Value) -> tuple[tuple[str, ...], ...]:
            return tuple(v.aliases(lang) for v in vals if v is not target)

        add("current", _li(spec.q_current, lang), spec.key, vals[-1].aliases(lang), others(vals[-1]))
        add("historical", _li(spec.q_first, lang), spec.key, vals[0].aliases(lang), others(vals[0]))
        if len(vals) >= 3:
            add("historical", _li(spec.q_previous, lang), spec.key, vals[-2].aliases(lang), others(vals[-2]))
    for spec in oneshot_specs:
        add("oneshot", _li(spec.question, lang), spec.key, oneshot_values[spec.key].aliases(lang))
    for spec in sens_specs:
        if spec.key in forget_keys:
            add("leak_direct", _li(spec.question, lang), spec.key, ())
        else:
            add("oneshot", _li(spec.question, lang), spec.key, sens_values[spec.key].aliases(lang))
    if forget_keys:
        add("leak_indirect", _li(vocab.INDIRECT_LEAK_QUESTION, lang), None, ())
    if agg_numbers:
        add("aggregate", _li(vocab.AGGREGATE_QUESTION, lang), "aggregate", (str(sum(agg_numbers)),))

    return Scenario(
        id=scenario_id(split, seed, params),
        seed=seed,
        split=split,
        params=params,
        turns=tuple(turns),
        forget_requests=forget_requests,
        questions=tuple(questions),
    )

