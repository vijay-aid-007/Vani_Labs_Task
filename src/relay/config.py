"""Central configuration for The Relay.

Every tunable number lives here. ``None`` means "not decided / not measured
yet". Values that come from the real service are filled in ONLY from probe
output (results/raw/) or docs pasted by the owner; never guess them.
Secrets are never stored here: only the NAME of the env var is.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
PAPERS_DIR: Path = DATA_DIR / "papers"
CHECKSUMS_FILE: Path = PAPERS_DIR / "CHECKSUMS.txt"
EVAL_DIR: Path = DATA_DIR / "eval"
CALIBRATION_FILE: Path = DATA_DIR / "calibration.json"
RESULTS_DIR: Path = ROOT / "results"
RAW_DIR: Path = RESULTS_DIR / "raw"
RESULTS_CSV: Path = RESULTS_DIR / "results.csv"
NOTES_FILE: Path = ROOT / "NOTES.md"

# --- Reproducibility -------------------------------------------------------
SEED: int = 42


# --- Library (the only knowledge source) -----------------------------------
@dataclass(frozen=True)
class Paper:
    """One arXiv paper pinned to an exact version."""

    arxiv_id: str
    version: int

    @property
    def url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}v{self.version}"

    @property
    def filename(self) -> str:
        return f"{self.arxiv_id}v{self.version}.pdf"


PAPERS: tuple[Paper, ...] = (
    Paper("1706.03762", 7),
    Paper("2410.00037", 1),
    Paper("2507.07955", 2),
)

# --- Step 0: download ------------------------------------------------------
DOWNLOAD_TIMEOUT_S: float = 60.0
DOWNLOAD_DELAY_S: float = 3.0  # pause between arXiv downloads (politeness)
MIN_PDF_BYTES: int = 50_000  # sanity floor, not a measurement
PDF_EOF_SCAN_BYTES: int = 2048  # where to look for the %%EOF marker
USER_AGENT: str = "relay-takehome/0.1"

# --- Retrieval (step 3) ----------------------------------------------------
EMBEDDING_MODEL: str | None = None
CROSS_ENCODER_MODEL: str | None = None
CHUNK_SIZE_CHARS: int | None = None
CHUNK_OVERLAP_CHARS: int | None = None
TOP_K: int | None = None
RERANK_ENABLED: bool | None = None
HYBRID_ALPHA: float | None = None  # weight of embeddings vs BM25
SCORE_THRESHOLD: float | None = None  # refusal threshold (decided in code)
RETRIEVAL_SETTINGS: dict[str, dict[str, object]] = {}  # >=3 named settings

# --- Pipeline (step 4) -----------------------------------------------------
MAX_REDRAFTS: int = 1  # brief: redraft once, then refuse. Hard cap.
TOOL_TIMEOUT_S: float | None = None

# --- Real model: Groq (step 1) ---------------------------------------------
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV_VAR: str = "GROQ_API_KEY"
API_KEY_PREFIX: str = "gsk_"  # my assumption of Groq's key format; fix if your real key differs
GROQ_MODEL: str | None = "openai/gpt-oss-20b"  # replaces retired llama-3.1-8b-instant (404 model_not_found)
LLM_TEMPERATURE: float | None = None
LLM_TIMEOUT_S: float = 60.0  # per-read httpx timeout; revisit from probe data

# --- Real-service limits: from probe output / pasted docs ONLY -------------
RPM_LIMIT: int | None = 30
TPM_LIMIT: int | None = 8000  # header x-ratelimit-limit-tokens, stage A, 2026-11-12
RPD_LIMIT: int | None = 1000  # header x-ratelimit-limit-requests, stage A, 2026-11-12
TPD_LIMIT: int | None = 200_000
MAX_IN_FLIGHT: int | None = None

# --- Step 1: probe design (experiment parameters, NOT service limits) -------
PROBE_SAFETY_FRACTION: float = 0.5  # pace stages B/C to at most this share of RPM/TPM
PROBE_A_MAX_TOKENS: int = 512  # generous: a reasoning model may spend tokens before the answer
PROBE_B_MAX_TOKENS: tuple[int, ...] = (256, 384, 512)
PROBE_B_PROMPT_WORDS: tuple[int, ...] = (20, 1500)
PROBE_B_REPS: int = 3
PROBE_C_LEVELS: tuple[int, ...] = (1, 2, 4, 8)  # concurrent requests per burst
PROBE_C_MAX_TOKENS: int = 64
PROBE_C_COOLDOWN_S: float = 5.0
PROBE_D_MAX_TOKENS: int = 8
PROBE_D_EXTRA_REQUESTS: int = 5  # sent beyond RPM_LIMIT while provoking a 429
PROBE_D_MAX_POLLS: int = 30
PROBE_D_POLL_INTERVAL_S: float = 1.0
TOKENS_PER_WORD_ESTIMATE: float = 1.5  # planning/pacing only; real counts come from usage

# --- Stand-in service (step 2). Fitted params live in CALIBRATION_FILE ------
STANDIN_HOST: str | None = None
STANDIN_PORT: int | None = None

# --- Load policy (step 5) --------------------------------------------------
GIVEN_UP_AFTER_S: float | None = None  # "N": justified from measured data
REQUEST_DEADLINE_S: float | None = None
QUEUE_MAX_LEN: int | None = None
BUCKET_SAFETY_MARGIN: float | None = None
LADDER_THRESHOLDS: dict[str, float] = {}  # per-rung trigger, set from data

# --- Retries (step 5) ------------------------------------------------------
RETRY_BASE_S: float | None = None
RETRY_CAP_S: float | None = None
RETRY_PER_REQUEST_CAP: int | None = None
RETRY_GLOBAL_BUDGET: int | None = None
NAIVE_RETRY_MODE: bool = False  # flag: immediate retries, for the experiment   