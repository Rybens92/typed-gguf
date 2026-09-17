# ggufone — thread (E1: silnik decyzji na GGUF, „Jev-like", bez fine-tuningu)

Repo roboczo: `/home/rybens/workspace/ggufone` (MIT). Nazwa robocza — zmiana przed publikacją = 1 commit.

## 2026-09-17 15:49 — @bots-coordinator — KICKOFF

**Cel:** lokalny silnik „typed decisions" dla dowolnego modelu GGUF (to, co obsługuje llama.cpp):
`state + pytania typowane → odpowiedzi typowane + prawdopodobieństwa/pewność`, w jednym przebiegu
na współdzielonym prefiksie (fork KV/stanu), bez generowania tekstu i **bez fine-tuningu modelu**.

**Zweryfikowane fakty (recon 2026-09-17):**
- Mechanika działa na modelach stock/zamrożonych — potwierdzone publicznie: `harshatheg/Qwen-2.5-1B-RLCD`,
  `rorshopping/jev-on-a-laptop` + `parallel-decisions`, `TheoLeeCJ/openjev`, `bnsd55/openjev`.
  Punkt odniesienia: stock 7B ≈ 73.8% vs Jev 86.6% (public eval TypeSafe); frozen Qwen3.5-4B ≈ 0.845 vs 0.883.
- llama.cpp C API ma wszystko: `n_seq_max`, `llama_memory_seq_cp` (fork, w tym pamięć rekurencyjna hybryd),
  `llama_get_logits_ith`, `llama_state_seq_save_file`, `llama_batch_init`.
  llama-cpp-python to wystawia; **nie** wystawia: `chat_template_kwargs`/`enable_thinking`, `n_cpu_moe`
  (ale `tensor_buft_overrides` są → offload ekspertów MoE wykonalny), `fit`.
- Test na żywo (ta maszyna): hybrydowy Qwen3.5-0.8B ładuje się; odczyt kandydatów przez `n_probs` +
  restricted softmax działa; prefill ~2k tok = 5.9 s CPU / 0.43 s Vulkan; decyzja z cache ≈ 14–20 ms;
  pierwszy strzał na Vulkanie ≈ 23–30 s (kompilacja shaderów → wymagany warm-up).
- Model domyślny: `XHToken/Spark-X2.5-4B-GGUF:Q8_0` (4.38 GB, Apache-2.0). Arch `spark2_5` wymaga
  llama.cpp ≥ **b10828**; vendored w llama-cpp-python (2026-09-04) i lokalne buildy (b10679/b10715) — za stare.
  Oficjalne binarki llama.cpp: build **b11026** (2026-09-17) z wariantami ubuntu-vulkan-x64, ubuntu-cuda-12.8/13.3,
  ubuntu-rocm-10.0, sycl, openvino, win-cuda, macos-arm64.
- Format docelowy: kształty TypeSafe (choice/score/noul + probabilities + confidence + usage) — adapter `--format typesafe`.
- Dev box: gcc 16.2, cmake 4.4, uv 0.12.5; brak nvcc/glslc/ninja; GPU RTX 3060 Ti 8 GB (Vulkan działa, CUDA w buildzie Unsloth nie wstał).

**Wymóg produktowy:** użytkownik **nigdy nie kompiluje**. Dystrybucja:
1) prebuilt: nasze wydania (GitHub Releases) — `ggufone init` wykrywa sprzęt i instaluje;
2) fallback: zautomatyzowany build ze źródeł z postępem;
3) ostatnia instancja: instrukcje ręczne.
Plan A: cienki pakiet Python + pobierane oficjalne biblioteki llama.cpp (pinned, do potwierdzenia że tarballe zawierają `.so`).
Plan B: nasze wheels budowane w CI z przypiętym submodulem llama.cpp.

**Zadania:** `t_7bcff796` (code-spec, SPEC + scaffold, workspace `/home/rybens/workspace/ggufone`).
Kolejne etapy: E1a runtime+modele, E1b silnik (fork/logity/schema), E1c reasoning-resolver + fit,
E2 benchmark (Vulkan/CUDA/CPU), E2.5 auto-kalibracja + routing, E3 Qwen3.6-35B-A3B / Qwen3.8-27B.

**Otwarte:** nazwa projektu (roboczo `ggufone`).

## 2026-09-17 15:55 — @bots-coordinator — ROZSTRZYGNIĘTE: dystrybucja bez kompilacji (Plan A potwierdzony empirycznie)

Sprawdzone na oficjalnym wydaniu `b11026` (2026-09-17):
- paczki llama.cpp **zawierają biblioteki współdzielone**: `libllama.so`, `libggml.so`, `libggml-base.so`,
  `libggml-cpu-*.so` oraz binarki (`llama-server`, `llama-fit-params`, `llama-tokenize`);
- rozmiary: **16 MB** (ubuntu-x64 CPU), **28 MB** (ubuntu-vulkan-x64); są też warianty cuda-12.8/13.3, rocm-10.0, sycl, openvino, macos, windows;
- **test ctypes na tej maszynie: 21/21 potrzebnych funkcji obecnych** w `libllama.so`
  (model/context params, load, tokenize, meta, n_seq_max, batch, decode, get_logits_ith,
  memory_seq_cp/rm, state_seq_save/load, chat_apply_template, n_layer, vocab);

**Decyzja:** silnik główny = **ctypes bezpośrednio do `libllama.so`** z przypiętego oficjalnego wydania
(`ggufone init` pobiera pasujący wariant 16–80 MB; zero kompilacji po stronie użytkownika).
`llama-cpp-python` = opcjonalny backend zgodnościowy (jego vendored llama.cpp jest za stary na `spark2_5`,
a wheels nie mają Vulkanu). Nasze własne wheels (CI) pozostają jako fallback dla platform bez oficjalnych paczek.
`llama-fit-params` z tej samej paczki służy do auto-dopasowania parametrów do sprzętu.
Pin wersji runtime w locku (<repo>/runtime.lock) + pre-flight check arch modelu vs runtime.

## 2026-09-17 16:05 — @bots-coordinator — ✅ PoC ctypes DZIAŁA (dowód end-to-end na hybrydowym Qwen3.5-0.8B)

Referencyjna implementacja: `state/groupchat/poc-ctypes-20260917.py` (uruchamialna, ~210 linii, zero zależności poza stdlib).

Wyniki na libllama.so z oficjalnego wydania **b11026** (paczka ubuntu-x64, 16 MB), model `Qwen3.5-0.8B-UD-Q4_K_XL`:
```
[1] model załadowany 0.67 s   arch=qwen35  layers=24  vocab=248320
[3] prefill 44 tok  →  279 ms (CPU)
[4] fork: seq_cp(0→1), seq_cp(0→2)  — na hybrydzie (KV + stan rekurencyjny)
[5] JEDEN batched decode dwóch gałęzi (63 tok)  →  397 ms
[6] restricted softmax:  gałąź1 {A .295 B .291 C .266 D .148} | gałąź2 {B .494 A .410 …}
[7] fork vs sekwencyjnie: max |Δ| = 0.00e+00   → izomorfizm potwierdzony
```

**Trzy pułapki odkryte i rozwiązane (dla implementacji):**
1. **Backendy trzeba załadować jawnie**: `ggml_backend_load_all_from_path(<dir z libs>)` — bez tego `llama_model_load_from_file` zwraca NULL
   („no backends are loaded").
2. **`kv_unified = true` w kontekście jest WYMAGANE** dla taniego forkingu: bez tego każda sekwencja ma własny strumień KV,
   a `seq_cp` między strumieniami wymaga pełnego zakresu (`GGML_ASSERT(is_full)`) i wywala proces.
   Z `kv_unified=true` fork = dopisanie seq_id do współdzielonych komórek (tak robi to też serwer llama.cpp).
3. **Indeksowanie logitów**: `llama_get_logits_ith(ctx, i)` używa **indeksu tokenu w batchu**, nie kolejności flag `logits`.

Wszystkie trzy pozycje są już uwzględnione w PoC → pipeline ma działający wzorzec do E1b.

## 2026-09-17 16:20 — @bots-coordinator — SPEC zweryfikowany pierwszej ręki (nie tylko z raportu)

Artefakty w repo potwierdzone: commit `d5241fd` (SPEC.md 751 linii, `runtime.lock`, `docs/verify_runtime_contract.py`,
`docs/evidence/*` z moim PoC, workflows CI/matrix/wheels, scaffold `src/ggufone/*`, LICENSE MIT).

Uruchomione przeze mnie na tej maszynie:
- `python3 docs/verify_runtime_contract.py` → **exit 0, failures: 0, skips: 2** (skip B: brak zainstalowanego runtime — E1a
  uruchamia tę sekcję; skip D: mirror readout — E1b). Oracle potwierdził m.in. piny wydania i assetów, 61 wpisów w tarballu,
  piny linii nagłówka, SHA modelu domyślnego, obecność markerów PoC i **3 udokumentowane pułapki**; odtworzył liczby
  referencyjne (softmax, `confidence` vs wartości z dokumentacji TypeSafe z jawnym „no parity claim", `score=Σi·pᵢ`,
  coverage, KV/token/seq dla `spark2_5` = 147456 B f16, rekomendację quantu dla 8 GiB).
- `uv run pytest -q` → **30 passed**.

Status: **SPEC DRAFT gotowy; E1a czeka na ratyfikację S-1..S-12 (§8) przez operatora** — to jedyna bramka wejściowa
do E1a (poza tym entry conditions spełnione: oracle zielony, PoC w repo, struktury ctypes zweryfikowane).

## 2026-09-17 16:55 — @bots-coordinator — S-1..S-12 RATYFIKOWANE (operator: „wszystko jest dobrze"); E1a wystawione

- Operator zaakceptował **wszystkie** S-1..S-12. Kolejka **naturalna** (bez podnoszenia priorytetu): kolejne zadania
  wchodzą na tablicę `coding` i czekają na wolnych workerów.
- **Audyt kompletności** (pytanie operatora „czy o czymś nie zapomnieliśmy?") → 4 drobne luki dołączone do E1a jako
  „coordinator additions”: (a) obsługa HF auth dla repo gated/private (`HF_TOKEN`, `E_HF_AUTH_REQUIRED`),
  (b) precheck wolnego miejsca na dysku (`E_INSUFFICIENT_DISK`), (c) pole `license` w registry + drukowanie przy
  `models pull`/`ls`, (d) `NOTICE`/credits dla upstreamu (llama.cpp MIT + harshatheg/rorshopping/TheoLeeCJ/bnsd55 + docs TypeSafe).
- Odłożone (nice-to-have, nie blokują): `model_sha256`/`prompt_hash` w odpowiedzi native, linia o polityce współbieżności
  `serve`/`mcp`, procedura aktualizacji pinu runtime w docs.
- **Zadanie:** `t_541fdedb` — „ggufone — E1a: runtime + model registry (init/doctor/models, no engine yet)", assignee `code-tdd`,
  priorytet 12, workspace `/home/rybens/workspace/ggufone`, status `ready`.

## 2026-09-17 18:35 — @bots-coordinator — WERYFIKACJA E1a: NIE domknięte na hoście → FIX `t_eae35404`

E1a zgłoszone jako done (commity `84cb0c1` / `210da65` / `a4ab12e`). Weryfikacja pierwszej ręki **na tej maszynie**:

- `uv run pytest -q` → **7 failed, 254 passed, 11 skipped**. Worker walidował w **sandboxie podman BEZ GPU** —
  sam to zapisał w `docs/evidence/e1a_baseline.json` („sandbox … NOT the host … no /dev/dri, no nvidia-smi").
- **Przyczyna źródłowa (jedna):** warstwa detekcji czyta prawdziwy host **nawet przy wstrzykniętych `probes`** →
  `detect_backend({"system":"linux","dri_nodes":["/dev/dri/renderD128"]})` zwraca `cuda` (test oczekuje `vulkan`);
  `host_variant("auto", linux, x86_64)` zwraca `linux-x64-cuda-12.8` (test oczekuje `linux-x64-cpu`); 5 testów CLI
  zależy od tego samego mapowania. Na maszynie bez GPU testy przechodzą — na tej nie.
- `ggufone init --dry-run --json` na hoście: wariant `linux-x64-cuda-12.8`, 168.8 MB, `required_bytes` ~506 MB (poprawnie dla tej karty).
- `ggufone doctor --json` na hoście: `status: failures`, exit 1 (brak zainstalowanego runtime — oczekiwane przed `init`).
- **Dodatki operatora: wszystkie 4 obecne** — HF auth (`E_HF_AUTH_REQUIRED`), precheck dysku (`E_INSUFFICIENT_DISK`),
  pole `license` w registry (`store.py` ENTRY_FIELDS + `hf.py`), credits w `README.md` („Credits and attribution": llama.cpp, TypeSafe, rorshopping…).
- **Bramka live (A-E1a-1, oracle bez SKIP) nieudowodniona na hoście** — brak zainstalowanego runtime; w sandboxie raportowana jako zielona.

→ **FIX: `t_eae35404`** (code-tdd, priorytet 12, kolejka naturalna): probes w pełni wstrzykiwalne + testy w dwóch światach
(GPU brak/jest), suite zielony offline na hoście, pełna bramka live na tej maszynie (`init` → `doctor` → oracle **bez SKIP** →
`pytest --run-network` + poisoned PATH ≤180 s), fallback `cuda→vulkan→cpu` z dowodem, sekcja „host" w `e1a_baseline.json`.

## 2026-09-17 16:40 — @code-spec — SPEC + SCAFFOLD + ORACLE GOTOWE (t_7bcff796)

**Deliverable:** `SPEC.md` (root), `docs/verify_runtime_contract.py` (oracle — exit 0), `docs/evidence/`
(7 artefaktów wykonanych), `runtime.lock` (pin runtime'u), scaffold (`pyproject.toml` uv, `src/ggufone/`
= 32 moduły, `tests/`, 3 workflowy w `.github/workflows/`), `README.md`, `LICENSE` (MIT).

**Gate'y policzone WYKONANIEM na tej maszynie:**
```
uv run pytest -q        → 30 passed   (E0: import wszystkich modułów, CLI, katalog błędów, brak treningu)
uv run ruff check .     → All checks passed
python3 docs/verify_runtime_contract.py                              → exit 0 (offline; 2 SKIP)
GGUFONE_RUNTIME_DIR=/tmp/ggufone_probe/rel/llama-b11026 python3 docs/verify_runtime_contract.py → exit 0 (live; 1 SKIP = funkcje E1b)
```
Scaffold przetestowany od zera: `uv sync --extra dev` (venv + pytest/ruff) — działa.

**Co oracle dowodzi wykonaniem (nie prozą) — pełna lista w `SPEC.md` §6:**
33 assety b11026 + rozmiary 9 używanych; sha256 tarballa ubuntu-x64 + 61 wpisów (m.in. `libllama.so`,
`libggml.so`, `llama-fit-params`); **32/32 symbole `libllama.so` + 2/2 `libggml.so` przez ctypes**;
`spark2_5` obecny w `libllama.so`; `llama-cli --version` = `0.4.1-dev (build 11026, commit b49650adb)`
(uwaga: banner idzie na **stderr**); pin HF `902d8659…`, Q8_0 = 4 375 021 152 B, sha256 `5c2c3c19…`;
`lfs.oid == sha256(plik)` sprawdzone na realnym obiekcie LFS; lokalny `Spark-X2.5-4B-Q8_0.gguf`
hashuje się dokładnie do pinu; nagłówki GGUF (Spark: `spark2_5`, 36 warstw, 4 kv-heady, key_len 256 →
**147 456 B KV/token/fork f16**, q8_0 = 73 728 B); formuły (softmax, confidence=znormalizowany pik,
score=Σi·p_i=1.30, coverage, `recommend_quant` 8 GiB → Q8_0 + KV q8_0 = 7.33 GB).

**Milestone'y + AC (SPEC §5, skrót):**
- **E1a** runtime+modele: `init` bez kompilatora na PATH (A1), `doctor` (symbole/build/backend), `models pull`
  z resume+sha256+atomowym `os.replace`, `recommend-quant` wg tabeli executed, pre-flight arch
  (`E_MODEL_ARCH_UNSUPPORTED` zamiast crashu), baseline A13. Bramka: oracle live **bez SKIP**.
- **E1b** silnik: schemat+walidacja (E_*), prefill raz + forki + fale (n_seq_max), readout sequence/single_token,
  coverage/reliability (`W_LOW_MASS`, nigdy ciche renormalizowanie), save/load stanu, determinizm
  (threads=1 → identyczny JSON), CLI `run`/`ask`, adapter typesafe (dokładne klucze z docs), zakaz
  samplowania (grep `llama_sampler_` + spy na liczbę `llama_decode`). Bramka: fork vs sekwencyjnie ≤1e-3.
- **E1c** reasoning-resolver + fit: łańcuch szablonów (GGUF `chat_template` → `llama_chat_apply_template`
  → override → `E_TEMPLATE_UNRESOLVED`), wygaszanie thinkingu, `ggufone fit` = `llama-fit-params`
  + nasza matematyka KV/n_seq_max + cache per (sha modelu, host), `docs/TEMPLATES.md`, E2E na Spark-X2.5.
- **E2** benchmarki: suites latency/throughput/quality/calibration/determinism, dev set ≥50 pozycji,
  ponowny pomiar liczb reconu (5.9 s CPU / 0.43 s Vulkan / 14–20 ms / warm-up 23–30 s) side-by-side.
- **E2.5** kalibracja + routing: temperatury per (model, typ), akceptacja tylko gdy poprawia ECE,
  routing budżetowy + eskalacja opt-in (max 1).
- **E3** 27–35B: 27B + 35B-A3B na 8 GB VRAM / 31 GB RAM, MoE offload, 32k-token smoke, tabela jakości vs 4B.

**Do ratyfikacji (SPEC §8, S-1..S-12):** nazwa `ggufone`; ctypes/official primary (rozstrzygnięte przez
operatora — zapisane); confidence = znormalizowany pik **bez deklaracji parity** (7/8 przykładów z docs
w ±0.02; ósmy — quickstart 0.596 vs nasze 0.760 — udokumentowany jako outlier, bo nie pasuje do żadnej
statystyki rozproszenia); konserwatywny planner pamięci do pomiaru A-E1a-8; readout=sequence,
`length_norm=1.0`; `kv_type=auto` (f16→q8_0→q4_0); eskalacja off do E2.5; HTTP `127.0.0.1:8088`; nazwy MCP;
dev set własny ≥50; E2 report-only (bez progu parity); `state_cache` on / `save_state` off.

**NEXT:** ratyfikacja S-1..S-12 → `code-tdd` start **E1a**: pierwszy test do zzielenienia = oracle live
bez SKIP; transplantacja `docs/evidence/poc-ctypes-20260917.py` → `src/ggufone/runtime/ctypes_binding.py`
(struktury 1:1, pole w polu); twarde zasady: zero zależności runtime poza stdlib, zero kompilatora
w `ggufone init`, żadnych binarek w git.

## 2026-09-17 17:35 — @code-tdd — E1a FIX (t_eae35404): przyczyna naprawiona, fallback z dowodem; bramka HOST do uruchomienia

Commity: `f3ae67d` (czystość sond), `6ef79d2` (fallback cuda→vulkan→cpu + doctor) + commit
z bramką hosta i evidence (`tools/host_gate_e1a.{sh,}`, `docs/evidence/e1a_baseline.json`).

**1. Przyczyna źródłowa naprawiona w implementacji (zero zmian oczekiwań testów).**
`runtime/pins.py`: detekcja jest teraz czystą funkcją obiektu `HostProbes`; `current_host()`
to jedyne miejsce w pakiecie, które czyta maszynę (`platform.system/machine`,
`shutil.which("nvidia-smi")`, `DRI_DIR.glob("renderD*")`, `ICD_DIR`). Podanie **jakiejkolwiek**
sondy (`system`/`machine`/`has_nvidia_smi`/`dri_nodes`/`icd_dir` lub `probes=`) przełącza na
świat syntetyczny, w którym niepodane fakty są NIEOBECNE — żaden odczyt hosta nie może wyciec.
Produkcja bez zmian: `init --dry-run` przy obecnym `nvidia-smi` → `linux-x64-cuda-12.8`
(168 811 114 B). Test-tripwire wywala każdy wyciek (`shutil.which`/`platform.*`/`/dev/dri`/ICD).

**2. Suite w OBU światach (ten sam kontener bez GPU):**
```
uv run pytest -q                                    -> 281 passed, 11 skipped
PATH=<fałszywy nvidia-smi> uv run pytest -q         -> 281 passed, 11 skipped
HOME=<z zainstalowanym runtime> uv run pytest -q    -> 280 passed, 10 skipped (sekcja B na żywo)
HOME=<...> uv run pytest -q --run-network           -> 289 passed, 1 skipped (brak Qwen3.5-0.8B)
```
Zmiana w testach CLI: autouse-fixture wskazuje jawną, fałszywą maszynę CPU — asercje pozostały
identyczne (to były testy planu wariantu, nie maszyny, na której akurat biegną).

**3. Fallback z PRAWDZIWYM dowodem (wymaganie 4).** Głęboka sonda dlopen-uje każdy
`libggml-<backend>`; na tym sandboxie prawdziwy `libggml-cuda.so` z pinu b11026 **nie ładuje
się**: `libcudart.so.12: cannot open shared object file` (`ldd`: cudart/cublas/libcuda =>
not found). `init` (auto) zszedł więc realnie **cuda → vulkan**: pobrał 30 294 625 B
(rozmiar = pin), zainstalował `linux-x64-vulkan`, zapisał `fallback_reason`, a `doctor --json`
raportuje `working_backend: vulkan` + listę backendów + check `runtime.fallback`. Odrzucony
wariant jest usuwany z dysku (inaczej `find_runtime()` łapał `...cuda-12.8` alfabetycznie i
`doctor` sondował zepsuty bundle). `runtime.json` trzyma: `backend_requested`,
`backend_working`, `backend_errors`, `fallback_attempts`, `fallback_reason`.

**4. Bramka live na HOŚCIE — jedyna rzecz, której nie wykonam** (sandbox: brak `/dev/dri`,
brak `/dev/nvidia*`, brak `nvidia-smi`). Przygotowane i przetestowane u mnie end-to-end:
```
bash tools/host_gate_e1a.sh                  # init → doctor → oracle → pytest --run-network → poisoned PATH
python3 tools/host_gate_summary.py <logdir>  # podsumowanie JSON do wklejenia do e1a_baseline.json
```
W logdir: surowe wyjścia, exit code'y, czasy, sha256 pobranych assetów, `runtime.json`.
Przebieg u mnie (fałszywy `nvidia-smi`, prawdziwe downloady): init exit 0 (2 s) → wariant
vulkan po odrzuceniu cuda; **oracle exit 0 z 0 SKIP w sekcji B**; `--run-network` exit 0;
poisoned PATH exit 0 (2 s, **0** wywołań shimów kompilatora); `doctor` exit 2 (warningi:
fallback + brak modelu w registry).

**Prośba do @bots-coordinator:** uruchom `tools/host_gate_e1a.sh` na tej maszynie i wklej
podsumowanie (jest w `host_gate_e1a.json`). Jeśli `libggml-cuda.so` nie załaduje się tam
(brak `libcudart` w systemie), bramka zapisze `variant=linux-x64-vulkan` + `fallback_reason`:
to **PASS dla wymagania 4**, a finding dla „cuda-12.8 per dry-run” z wymagania 3 — dokładnie
ta informacja, której brakowało w poprzednim raporcie.

**Tier:** karta nie deklaruje tieru → domyślny M; mutacja (mutmut 3.8) na modułach zmienionych,
wynik w `docs/evidence/e1a_qa.md` + `e1a_baseline.json`. Task zablokowany do czasu uruchomienia
bramki hosta (nie domykam karty twierdzeniem, którego nie mam czym pokryć).

