"""
decision.py
===========
The decision layer, behind one interface so a strategy can be run three ways
without changing a line of the strategy itself.

  RuleDecider   take every candidate the strategy proposes. The naive baseline.
  GateDecider   same candidates, vetoed by hand-written context filters.
  ModelDecider  same candidates, scored by a decision model on the same context.

The gated arm exists to keep the comparison honest. Any filter that removes
trades will raise a win rate, so "the model lifted the win rate" means nothing
on its own. The question worth answering is whether the model beats three
if-statements.

The model is Laya (huggingface.co/convaiinnovations/laya, Apache-2.0), an open
System One decision model served locally by `laya-serve`. It speaks the
/v1/systemone wire protocol TypeSafe Jev defined, so the same client serves
both: the `laya` arm reads MODEL_URL / MODEL_NAME, the `jev` arm reads JEV_URL /
JEV_MODEL / TYPESAFE_API_KEY (see PROVIDERS):
  request : {"model","state","questions":{id:{"type","instructions","criteria"}}}
  response: {"model","answers":{id:{"type","choice","confidence","probabilities"}},"usage"}
  a "score" question returns the probability-weighted index of its criteria list.
Laya adds POST /v1/systemone/batch ({"states": [...]} -> {"results": [...]}),
which prefetch() uses, and reports `usage.truncated` when a state was cut to fit
the checkpoint's context window.

We speak HTTP with stdlib urllib rather than importing `laya`, which needs
Python >= 3.10 and torch. The model runs as a sidecar; the bots stay small.

Every response is cached to cache/laya_cache.jsonl keyed by a hash of the
provider (laya / jev) and the exact request, so a re-run is free and returns
identical decisions. A black-box model has no business in a backtest you cannot reproduce.
"""

from __future__ import annotations

import hashlib
import http.client
import threading
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from contracts import Action, Decision, ENTRIES, Snapshot

DEFAULT_CACHE = Path(__file__).parent / "cache" / "laya_cache.jsonl"

# Where each named arm points by default: (url env, default url, model env,
# default model, api-key envs in order). Env vars are read at construction, not
# import, so a runner's load_env(.env) still applies. Keeping the two providers
# on separate env vars is what lets `--arms gated laya jev` compare them in one run.
PROVIDERS = {
    # laya checkpoints: english (512 tokens) | multilingual (1024) | typed-decisions (1024)
    "laya": ("MODEL_URL", "http://localhost:8000/v1/systemone",
             "MODEL_NAME", "typed-decisions",
             ("MODEL_API_KEY", "LAYA_API_KEY")),
    "jev": ("JEV_URL", "https://api.typesafe.ai/v1/systemone",
            "JEV_MODEL", "jev-latest",
            ("TYPESAFE_API_KEY",)),
}

# A gate takes a snapshot and returns a veto reason, or None to allow it.
Gate = Callable[[Snapshot], Optional[str]]


class Decider:
    name = "base"

    def decide(self, snap: Snapshot) -> Decision:
        raise NotImplementedError

    def stats(self) -> Dict:
        return {}


# --------------------------------------------------------------- arm 1

class RuleDecider(Decider):
    """Takes whatever the strategy proposes. This is the strategy 'as advertised'."""
    name = "rules"

    def decide(self, snap: Snapshot) -> Decision:
        return Decision(action=snap.proposed, source="rules")


# --------------------------------------------------------------- arm 2

class GateDecider(Decider):
    """Entry candidates must survive every gate. Exits and holds pass through."""
    name = "gated"

    def __init__(self, gates: List[Gate]):
        self.gates = gates
        self.vetoes: Dict[str, int] = {}
        self.allowed = 0

    def decide(self, snap: Snapshot) -> Decision:
        if snap.proposed not in ENTRIES:
            return Decision(action=snap.proposed, source="gated")

        for gate in self.gates:
            reason = gate(snap)
            if reason:
                self.vetoes[reason] = self.vetoes.get(reason, 0) + 1
                return Decision(action=Action.WAIT, confidence=0.0,
                                source="gated", note=f"veto:{reason}")

        self.allowed += 1
        return Decision(action=snap.proposed, source="gated")

    def stats(self) -> Dict:
        total = self.allowed + sum(self.vetoes.values())
        return {
            "candidates": total,
            "allowed": self.allowed,
            "vetoed": sum(self.vetoes.values()),
            "veto_breakdown": dict(sorted(self.vetoes.items(), key=lambda kv: -kv[1])),
        }


# --------------------------------------------------------------- arm 3

@dataclass
class ModelPrompt:
    """
    What to ask the model. Supplied by the strategy, because only the strategy
    knows what its own candidates mean.

    `entry_criteria` keys must be Action values (enter_long / enter_short / wait):
    the decider reads the probability of the proposed side by that key.

    `extra_questions` are asked alongside the decision and recorded but never
    acted on. They cost nothing extra (one call answers all questions in
    parallel) and let us check afterwards whether the model's stated risk
    actually predicted the outcome.
    """
    entry_instructions: str
    entry_criteria: Dict[str, str]
    extra_questions: Dict[str, dict] = field(default_factory=dict)
    manage_instructions: Optional[str] = None
    manage_criteria: Optional[Dict[str, str]] = None


def temper(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    """
    Soften (T > 1) or sharpen (T < 1) a distribution: p_i^(1/T), renormalised.

    Laya's own docs say its checkpoints are over-confident until fitted on your
    data. core/calibrate.py fits T on the in-sample half; this applies it.
    """
    if not probs or temperature == 1.0:
        return dict(probs)
    powered = {k: max(float(v), 1e-12) ** (1.0 / temperature) for k, v in probs.items()}
    total = sum(powered.values())
    return {k: v / total for k, v in powered.items()}


class ModelDecider(Decider):
    """
    A decision model as a second opinion on candidates the strategy already found.

    The model is not hunting for trades here. The strategy still proposes every
    entry; the model only answers "take it or stand aside". Both arms therefore
    see an identical candidate set, which is the only way the A/B means anything.
    """
    name = "laya"

    def __init__(
        self,
        prompt: ModelPrompt,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        name: str = "laya",
        threshold: float = 0.55,
        temperature: float = 1.0,
        max_len: Optional[int] = None,
        manage_positions: bool = False,
        cache_path: Path = DEFAULT_CACHE,
        offline: bool = False,
        timeout: float = 30.0,
        retries: int = 4,
        fallback: str = "wait",     # what to do when the model cannot answer: wait | rule
        log_path: Optional[Path] = None,   # append every decision for the dashboard
    ):
        self.prompt = prompt
        self.name = name
        url_env, url_default, model_env, model_default, key_envs = \
            PROVIDERS.get(name, PROVIDERS["laya"])
        self.url = url or os.environ.get(url_env) or url_default
        self.model = model or os.environ.get(model_env) or model_default
        # A local laya-serve without LAYA_API_KEY needs no key, so a missing key
        # is not an error here; a server that wants one answers 401 and we say so.
        self.api_key = api_key or next(
            (os.environ[k] for k in key_envs if os.environ.get(k)), "")
        env_len = os.environ.get("MODEL_MAX_LEN") if name != "jev" else None
        self.max_len = max_len or (int(env_len) if env_len else None)
        self.threshold = threshold
        self.temperature = temperature
        self.manage_positions = manage_positions
        self.cache_path = Path(cache_path)
        self.offline = offline
        self.timeout = timeout
        self.retries = retries
        self.fallback = fallback
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")      # fresh log per run

        self.cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()      # cache file + counters, for prefetch
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.below_threshold = 0
        self.disagreed = 0
        self.truncated = 0
        self._warned_context = False
        self._batch_route = True           # until a server says it has none
        self.latencies: List[float] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._load_cache()

    # -- cache -----------------------------------------------------

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with open(self.cache_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def _save(self, key: str, response: Dict) -> None:
        with self._lock:
            self.cache[key] = response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")

    def _key(self, payload: Dict) -> str:
        # The provider is part of the key, so the same question asked of Jev and of
        # Laya never answers from each other's cache. The URL is deliberately not:
        # a cache built against localhost:8000 must still serve the bot that
        # reaches the same model as laya:8000 inside docker. (The payload already
        # carries the checkpoint name.)
        blob = json.dumps({"provider": self.name, **payload}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    # -- transport -------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, payload: Dict, url: Optional[str] = None,
              timeout: Optional[float] = None) -> Dict:
        req = urllib.request.Request(
            url or self.url,
            data=json.dumps(payload).encode(),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def _with_retries(self, fn: Callable[[], Dict]) -> Optional[Dict]:
        """Run one HTTP call with backoff. Returns None when it cannot be answered."""
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except urllib.error.HTTPError as e:
                # 4xx other than rate-limit will not fix itself; stop retrying.
                # laya-serve answers 503 at LAYA_MAX_CONCURRENT, so 5xx does retry.
                code = e.code
                if code != 429 and 400 <= code < 500:
                    detail = ""
                    try:
                        detail = e.read().decode()[:200]
                    except Exception:
                        pass
                    self.errors += 1
                    print(f"[{self.name}] HTTP {code} {detail}")
                    return None
                if attempt == self.retries:
                    self.errors += 1
                    print(f"[{self.name}] giving up after {attempt + 1} tries: HTTP {code}")
                    return None
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                # A long sequential run WILL meet dropped keep-alives, resets and
                # truncated bodies. urllib does not wrap those in URLError, so a
                # narrower net lets them kill the whole backtest mid-flight.
                if attempt == self.retries:
                    self.errors += 1
                    print(f"[{self.name}] giving up after {attempt + 1} tries: "
                          f"{type(e).__name__}: {str(e)[:80]}")
                    return None
            time.sleep(0.5 * (2 ** attempt))
        return None

    def _record(self, resp: Dict, latency_ms: float) -> None:
        usage = resp.get("usage") or {}
        with self._lock:
            self.latencies.append(latency_ms)
            self.calls += 1
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            if usage.get("truncated"):
                self.truncated += 1
                if not self._warned_context:
                    self._warned_context = True
                    print(f"[{self.name}] WARNING: state truncated to fit {self.model}'s context "
                          f"(dropped {usage.get('state_tokens_dropped', '?')} tokens). Shorten "
                          f"context_lines or set MODEL_MAX_LEN: the model is deciding on less "
                          f"than the gates see.")

    def _ask(self, payload: Dict) -> Optional[Dict]:
        key = self._key(payload)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if self.offline:
            return None

        t0 = time.time()
        resp = self._with_retries(lambda: self._post(payload))
        if resp is None:
            return None
        self._record(resp, (time.time() - t0) * 1000)
        self._save(key, resp)
        return resp

    # -- parsing ---------------------------------------------------

    @staticmethod
    def _answers(resp: Dict) -> Dict:
        return resp.get("answers") or {}

    def _aux(self, resp: Dict) -> Dict[str, float]:
        """Flatten every non-decision answer into plain numbers we can analyse later."""
        out: Dict[str, float] = {}
        pre = self.name
        for qid, ans in self._answers(resp).items():
            if qid == "action" or not isinstance(ans, dict):
                continue
            kind = ans.get("type")
            if kind == "score" and ans.get("score") is not None:
                out[f"{pre}_{qid}"] = float(ans["score"])
            elif kind == "noul" and ans.get("noul") is not None:
                out[f"{pre}_{qid}"] = float(ans["noul"])
            elif kind == "choice":
                for opt, p in (ans.get("probabilities") or {}).items():
                    out[f"{pre}_{qid}_{opt}"] = float(p)
            if ans.get("confidence") is not None:
                out[f"{pre}_{qid}_conf"] = float(ans["confidence"])
        return out

    # -- decide ----------------------------------------------------

    def _questions_for(self, snap: Snapshot) -> Optional[Dict]:
        if snap.proposed in ENTRIES:
            q = {"action": {"type": "choice",
                            "instructions": self.prompt.entry_instructions,
                            "criteria": self.prompt.entry_criteria}}
            q.update(self.prompt.extra_questions)
            return q
        if snap.proposed == Action.HOLD and self.manage_positions and self.prompt.manage_criteria:
            return {"action": {"type": "choice",
                               "instructions": self.prompt.manage_instructions or "",
                               "criteria": self.prompt.manage_criteria}}
        return None

    def _payload(self, state: str, questions: Dict) -> Dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        if self.max_len:
            payload["max_len"] = self.max_len    # laya-only field; leave unset for Jev
        return payload

    def _fallback(self, snap: Snapshot, note: str) -> Decision:
        action = snap.proposed if self.fallback == "rule" else Action.WAIT
        return Decision(action=action, confidence=0.0, source=self.name, note=note)

    def decide(self, snap: Snapshot) -> Decision:
        questions = self._questions_for(snap)
        if questions is None:
            return Decision(action=snap.proposed, source=self.name, note="not_asked")

        state = "\n".join(snap.context_lines)
        payload = self._payload(state, questions)
        cached = self._key(payload) in self.cache

        resp = self._ask(payload)
        if resp is None:
            return self._fallback(snap, "no_answer")

        ans = self._answers(resp).get("action")
        if not isinstance(ans, dict) or "choice" not in ans:
            return self._fallback(snap, "unparsed_response")

        raw_probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        probs = temper(raw_probs, self.temperature)
        aux = self._aux(resp)
        conf = float(ans.get("confidence", 0.0))
        latency = 0.0 if cached else (self.latencies[-1] if self.latencies else 0.0)
        # Tempering never changes the argmax, so this is the model's own choice.
        raw = ans["choice"]
        chosen = Action(raw) if raw in Action._value2member_map_ else snap.proposed

        def d(action: Action, note: str = "") -> Decision:
            dec = Decision(action=action, probabilities=probs, confidence=conf,
                           source=self.name, latency_ms=latency, cached=cached,
                           note=note, aux=aux)
            self._log(snap, state, dec, raw_probs, resp)
            return dec

        if chosen in ENTRIES:
            # Argmax is not enough. A 34/33/33 split has an argmax too.
            if probs.get(chosen.value, 0.0) < self.threshold:
                self.below_threshold += 1
                return d(Action.WAIT, f"below_threshold({probs.get(chosen.value, 0.0):.2f})")
            # Never let the model flip the side. Fading the strategy is a different
            # strategy, and mixing the two would make the A/B unreadable.
            if chosen != snap.proposed:
                self.disagreed += 1
                return d(Action.WAIT, "disagreed_side")

        return d(chosen)

    def prefetch(self, snapshots, workers: int = 1, batch_size: int = 16,
                 verbose: bool = True) -> Dict:
        """
        Ask every question up front, then let the backtest read answers from cache.

        This is only legitimate because the questions are independent: the
        strategy emits at most one candidate per symbol per session, and the model
        is asked only about entries, always from a flat position. No answer can
        change another's input, so batched and sequential asking produce
        identical results. If a strategy ever asks the model to manage an open
        position, that stops being true and this must not be used.

        laya-serve's /v1/systemone/batch answers up to 64 states sharing one
        question set in one forward pass, so states are grouped by question set
        and sent `batch_size` at a time (max 64). Each answer is cached under the key the
        single-state request would have, so decide() reads it unchanged.
        batch_size=0 sends one request per state (e.g. against hosted Jev).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        groups: Dict[str, List[Dict]] = {}
        seen = set()
        for snap in snapshots:
            questions = self._questions_for(snap)
            if questions is None:
                continue
            payload = self._payload("\n".join(snap.context_lines), questions)
            key = self._key(payload)
            if key in self.cache or key in seen:
                continue
            seen.add(key)
            groups.setdefault(json.dumps(questions, sort_keys=True), []).append(payload)

        n = sum(len(v) for v in groups.values())
        if not n:
            if verbose:
                print(f"[{self.name}] prefetch: all {len(snapshots)} decisions already cached")
            return {"fetched": 0, "cached": len(snapshots)}

        step = min(batch_size, 64) if batch_size > 0 else 1
        jobs: List[List[Dict]] = []
        for payloads in groups.values():
            jobs.extend(payloads[i:i + step] for i in range(0, len(payloads), step))
        if verbose:
            print(f"[{self.name}] prefetch: {n:,} new decisions in {len(jobs):,} requests "
                  f"on {workers} workers -> {self.url} ({self.model})")

        t0 = time.time()
        done = 0
        last_report = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._ask_batch if len(j) > 1 else self._ask_one, j)
                       for j in jobs]
            for f in as_completed(futures):
                done += f.result()
                if verbose and done - last_report >= 500:
                    last_report = done
                    rate = done / max(0.001, time.time() - t0)
                    print(f"[{self.name}] prefetch {done:,}/{n:,} "
                          f"({rate:.1f} decisions/s, {self.errors} errors)")
        elapsed = time.time() - t0
        if verbose:
            print(f"[{self.name}] prefetch done: {done:,} decisions in {elapsed/60:.1f} min "
                  f"({done/max(0.001, elapsed):.1f}/s), {self.errors} errors, "
                  f"{self.truncated} truncated")
        return {"fetched": done, "elapsed_s": round(elapsed, 1), "errors": self.errors,
                "truncated": self.truncated}

    def _ask_one(self, payloads: List[Dict]) -> int:
        return int(self._ask(payloads[0]) is not None)

    def _ask_batch(self, payloads: List[Dict]) -> int:
        """One /batch request for payloads that share model, questions and max_len."""
        if self.offline:
            return 0
        if not self._batch_route:
            return sum(self._ask_one([p]) for p in payloads)
        body = {k: v for k, v in payloads[0].items() if k != "state"}
        body["states"] = [p["state"] for p in payloads]
        batch_url = self.url.rstrip("/") + "/batch"
        # On CPU a state with three questions costs several seconds, and a batch
        # answers all of them before it replies, so the wait has to scale with it.
        wait = self.timeout + 10.0 * len(payloads)
        t0 = time.time()
        try:
            resp = self._post(body, batch_url, wait)
        except urllib.error.HTTPError as e:
            if e.code not in (404, 405):
                resp = self._with_retries(lambda: self._post(body, batch_url, wait))
            else:
                # No batch route (hosted Jev): not an error, just a slower path.
                with self._lock:
                    if self._batch_route:
                        self._batch_route = False
                        print(f"[{self.name}] no /batch route at {self.url}; "
                              f"sending one request per state")
                resp = None
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            resp = self._with_retries(lambda: self._post(body, batch_url, wait))
        results = (resp or {}).get("results")
        if not isinstance(results, list) or len(results) != len(payloads):
            # A server without the batch route (hosted Jev) or a malformed reply:
            # fall back to one request per state rather than losing the run.
            return sum(self._ask_one([p]) for p in payloads)
        per = (time.time() - t0) * 1000 / len(payloads)
        for payload, r in zip(payloads, results):
            self._record(r, per)
            self._save(self._key(payload), r)
        return len(payloads)

    def _log(self, snap: Snapshot, state: str, dec: Decision,
             raw_probs: Optional[Dict[str, float]] = None,
             resp: Optional[Dict] = None) -> None:
        """
        The decision stream: every question asked and every answer, in order.
        This is what the dashboard renders, what you screen-record, and what
        core/calibrate.py fits the temperature on (from raw_probabilities).
        """
        if not self.log_path:
            return
        usage = (resp or {}).get("usage") or {}
        rec = {
            "timestamp": snap.timestamp.isoformat(),
            "symbol": snap.symbol,
            "price": snap.price,
            "proposed": snap.proposed.value,
            "state": state,
            "action": dec.action.value,
            "probabilities": dec.probabilities,
            "raw_probabilities": raw_probs if raw_probs is not None else dec.probabilities,
            "temperature": self.temperature,
            "confidence": dec.confidence,
            "aux": dec.aux,
            "note": dec.note,
            "model": self.model,
            "truncated": bool(usage.get("truncated", False)),
            "latency_ms": round(dec.latency_ms, 1),
            "cached": dec.cached,
            "taken": dec.action in ENTRIES,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def stats(self) -> Dict:
        lat = sorted(self.latencies)

        def pct(p: float) -> float:
            return round(lat[min(int(len(lat) * p), len(lat) - 1)], 1) if lat else 0.0

        return {
            "model": self.model,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "truncated": self.truncated,
            "vetoed_below_threshold": self.below_threshold,
            "vetoed_disagreed_side": self.disagreed,
            "latency_ms_p50": pct(0.50),
            "latency_ms_p95": pct(0.95),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# The old names, so strategies written against the Jev version keep importing.
JevPrompt = ModelPrompt
JevDecider = ModelDecider

DECIDERS = {"rules": RuleDecider, "gated": GateDecider,
            "laya": ModelDecider, "jev": ModelDecider}
