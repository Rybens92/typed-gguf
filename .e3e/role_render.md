## E3e — the role split per family (role_split + json_instructed)

- generated: 2026-09-19T21:06:05Z
- models: `/var/home/rybens/.hermes/models` · dev items rendered: 2
- status: checks-failed 0 · not-renderable 1 · refused 0 · rendered 4

| file | family | arch | status | prefix | tail(s) | dropped | checks |
|---|---|---|---|---|---|---|---|
| Accio-Lab_occamy-1.0-Q4_K_L.gguf | qwen35moe | qwen35moe | rendered | 481 | 327,323 | "" | 5/5 |
| Ling-3.0-tiny-Q5_K_M.gguf | None | bailingmoe3 | rendered | 511 | 346,342 | "" | 5/5 |
| Spark-X2.5-4B-Q8_0.gguf | spark2_5 | spark2_5 | rendered | 552 | 363,359 | "\n" | 5/5 |
| Ternary-Bonsai-2-27B-PTQ1_0.gguf | qwen35 | qwen35 | rendered | 481 | 327,323 | "" | 5/5 |
| Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf | qwen35moe | qwen35moe | not-renderable |  |  | "" | 0/0 |

### qwen35moe

- generation prompt + opener: `"e\": \"<exactly one candidate name>\"}<|im_end|>\n<|im_start|>assistant\n{\"choice\": \""`
- tail opens: `"<|im_start|>user\nQUESTION:\nWhich team owns this incident?\nCandidates:\n- billing:"`
- prefix ends: `"o deploy happened in the last 24 hours and the database is reachable.<|im_end|>\n"` (481 chars, `prefix + tail == render + opener`)
- state-only render drops 0 char(s) from the shared prefix: "" (the template's own trailing text)

### Ling-3.0-tiny-Q5_K_M.gguf

- generation prompt + opener: `"candidate name>\"}\n<|role_end|><role>ASSISTANT</role>\n<think></think>{\"choice\": \""`
- tail opens: `"<role>HUMAN</role>QUESTION:\nWhich team owns this incident?\nCandidates:\n- billing"`
- prefix ends: `" deploy happened in the last 24 hours and the database is reachable.<|role_end|>"` (511 chars, `prefix + tail == render + opener`)
- state-only render drops 0 char(s) from the shared prefix: "" (the template's own trailing text)

### spark2_5

- generation prompt + opener: `"ate name>\"}\n<\uff5cend\u2581of\u2581sentence\uff5c><\uff5cstart\u2581of\u2581sentence\uff5c><|Bot|></think>\n{\"choice\": \""`
- tail opens: `"<\uff5cstart\u2581of\u2581sentence\uff5c><|User|>QUESTION:\nWhich team owns this incident?\nCandidates"`
- prefix ends: `" happened in the last 24 hours and the database is reachable.<\uff5cend\u2581of\u2581sentence\uff5c>"` (552 chars, `prefix + tail == render + opener`)
- state-only render drops 1 char(s) from the shared prefix: "\n" (the template's own trailing text)

### qwen35

- generation prompt + opener: `"e\": \"<exactly one candidate name>\"}<|im_end|>\n<|im_start|>assistant\n{\"choice\": \""`
- tail opens: `"<|im_start|>user\nQUESTION:\nWhich team owns this incident?\nCandidates:\n- billing:"`
- prefix ends: `"o deploy happened in the last 24 hours and the database is reachable.<|im_end|>\n"` (481 chars, `prefix + tail == render + opener`)
- state-only render drops 0 char(s) from the shared prefix: "" (the template's own trailing text)

### qwen35moe

- **not-renderable**: E_TEMPLATE_UNRESOLVED: no template could render this prompt. The GGUF tokenizer.chat_template uses unsupported template construct '`set` without a value' at line 172 (_terse_core); libllama's built-in templates (llama_chat_apply_template) do not match it either. Fix: pass --template <builtin-name|path-to-jinja|plain> (options.template in a request), or use a model whose template is inside the supported subset (27 filters, if/for/set/macro, `is` tests).
- fallback: --template plain (role_split keeps the plain USER:/ASSISTANT: framing), or a live run, where the builtin bridge may render it

