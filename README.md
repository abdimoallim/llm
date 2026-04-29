### llm

A header-only C library for interacting with LLM providers through a unified API.

### Requirements

- C99^
- [libcurl](https://curl.se/libcurl/)
- pthreads

### Supported Providers

| Provider    | String       | Default Base URL                                          |
| ----------- | ------------ | --------------------------------------------------------- |
| OpenAI      | `openai`     | `https://api.openai.com/v1`                               |
| Anthropic   | `anthropic`  | `https://api.anthropic.com/v1`                            |
| Groq        | `groq`       | `https://api.groq.com/openai/v1`                          |
| Ollama      | `ollama`     | `http://localhost:11434/v1`                               |
| Together AI | `together`   | `https://api.together.xyz/v1`                             |
| Mistral     | `mistral`    | `https://api.mistral.ai/v1`                               |
| Cohere      | `cohere`     | `https://api.cohere.ai/v1`                                |
| Gemini      | `gemini`     | `https://generativelanguage.googleapis.com/v1beta/openai` |
| DeepSeek    | `deepseek`   | `https://api.deepseek.com/v1`                             |
| OpenRouter  | `openrouter` | `https://openrouter.ai/api/v1`                            |
| Perplexity  | `perplexity` | `https://api.perplexity.ai`                               |
| Fireworks   | `fireworks`  | `https://api.fireworks.ai/inference/v1`                   |
| vLLM        | `vllm`       | `http://localhost:8000/v1`                                |
| Custom      | `custom`     | _(set `base_url` manually)_                               |

Anthropic and Cohere use their native request formats. All other providers use the OpenAI-compatible chat completions format, including Gemini, DeepSeek and vLLM.

### Quick Start

```c
#include "llm.h"

int main(void) {
  llm_init();

  llm_client_t client;
  llm_client_init(&client);
  client.provider = LLM_PROVIDER_OPENAI;
  client.api_key  = "sk-...";
  client.model    = "gpt-4o";

  llm_message_t msg = llm_msg_user("What is 2 + 2?");

  llm_request_t req;
  llm_request_init(&req);
  req.messages       = &msg;
  req.messages_count = 1;

  llm_response_t *resp = llm_complete(&client, &req);

  if (resp->error == LLM_OK)
    printf("%s\n", resp->content);

  free(msg.content);
  llm_response_free(resp);
  free(resp);
  llm_client_free(&client);
  llm_cleanup();
  return 0;
}
```

Compile with:

```sh
gcc -o prog prog.c -lcurl -lpthread -lm
```

### Client Configuration

```c
llm_client_t client;
llm_client_init(&client);

client.provider                = LLM_PROVIDER_ANTHROPIC;
client.api_key                 = "sk-ant-...";
client.model                   = "claude-opus-4-5";
client.base_url                = NULL;        // use default
client.timeout_seconds         = 60L;
client.max_retries             = 5;
client.retry_delay_ms          = 500;
client.retry_delay_max_ms      = 20000;
client.retry_backoff_multiplier = 2.0;
client.retry_on_rate_limit     = true;
client.verify_ssl              = true;
client.verbosity               = 0;           // 0=off 1=info 2=debug 3=trace
client.proxy                   = "http://myproxy:8080";
client.org_id                  = "org-...";   // OpenAI only
client.project_id              = "proj-...";  // OpenAI only

llm_client_add_header(&client, "X-Custom", "value");
```

### Request Options

```c
llm_request_t req;
llm_request_init(&req);

req.messages            = msgs;
req.messages_count      = n;
req.model               = "gpt-4o-mini";  // override client model
req.max_tokens          = 2048;
req.temperature         = 0.7;
req.top_p               = 0.9;
req.top_k               = 40;
req.frequency_penalty   = 0.1;
req.presence_penalty    = 0.1;
req.system              = "You are a helpful assistant.";
req.json_mode           = true;
req.use_seed            = true;
req.seed                = 42;

char *stops[]     = {"STOP", "END"};
req.stop          = stops;
req.stop_count    = 2;
```

### Messages

```c
llm_message_t msgs[4];
msgs[0] = llm_msg_system("You are an expert.");
msgs[1] = llm_msg_user("Tell me about black holes.");
msgs[2] = llm_msg_assistant("Black holes are...");
msgs[3] = llm_msg_user("How are they formed?");

// Tool result message
llm_message_t tool_msg = llm_msg_tool_result("call_abc123", "{\"temp\": 72}");
```

Clean up message content when done: `llm_message_free(&msg)` or `free(msg.content)`.

### Streaming

```c
void on_chunk(const char *delta, bool done, void *user_data) {
  if (done) { printf("\n"); return; }
  if (delta) fputs(delta, stdout);
  fflush(stdout);
}

req.stream          = true;
req.stream_callback = on_chunk;
req.stream_user_data = NULL;

llm_response_t *resp = llm_complete(&client, &req);
```

Time-to-first-token is automatically measured and available in `resp->stats.time_to_first_token_ms`.

### Async (fire-and-forget)

```c
void on_done(llm_response_t *resp, void *user_data) {
  printf("Got: %s\n", resp->content);
  llm_response_free(resp);
  free(resp);
}

llm_complete_async(&client, &req, on_done, NULL);
// callback is called from a detached thread
```

### Batch (parallel requests)

```c
llm_request_t reqs[8];
for (int i = 0; i < 8; i++) {
  llm_request_init(&reqs[i]);
  reqs[i].messages       = &msgs[i];
  reqs[i].messages_count = 1;
}

// Run up to 4 concurrently, NULL callback = collect only
llm_response_t **results = llm_complete_batch(&client, reqs, 8, NULL, NULL, 4);

for (int i = 0; i < 8; i++) {
  if (results[i] && results[i]->error == LLM_OK)
    printf("[%d] %s\n", i, results[i]->content);
}

llm_batch_free(results, 8);
```

### Tool Calling

```c
llm_tool_def_t tools[1];
tools[0] = llm_tool_function(
  "get_weather",
  "Get current weather for a city",
  "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}"
);

req.tools       = tools;
req.tools_count = 1;
req.tool_choice = "auto";

llm_response_t *resp = llm_complete(&client, &req);

if (resp->tool_calls_count > 0) {
  printf("Tool: %s\n", resp->tool_calls[0].name);
  printf("Args: %s\n", resp->tool_calls[0].arguments_json);
}

llm_tool_def_free(&tools[0]);
```

### Response

```c
resp->id                   // request ID string
resp->model                // model name echoed by provider
resp->content              // text content (NULL if tool call only)
resp->finish_reason        // LLM_FINISH_STOP / LENGTH / TOOL_CALL / etc.
resp->tool_calls           // array of tool calls
resp->tool_calls_count
resp->error                // LLM_OK or error code
resp->error_message        // human-readable error string
resp->http_status          // raw HTTP status code
resp->usage.prompt_tokens
resp->usage.completion_tokens
resp->usage.total_tokens
resp->usage.provided       // whether the provider returned token counts
```

### Stats

Every response includes a `stats` field:

```c
resp->stats.latency_ms              // total wall time in ms
resp->stats.retries                 // number of retry attempts made
resp->stats.time_to_first_token_ms  // ms until first streamed token
resp->stats.has_time_to_first_token // false for non-streaming requests
resp->stats.stream_chunks           // number of SSE chunks received
```

Print all stats at once:

```c
llm_print_stats(resp, stderr);
```

Output format:

```
=== stats ===
latency_ms: 843.21
retries: 0
prompt_tokens: 24
completion_tokens: 118
total_tokens: 142
token_usage_provided: yes
time_to_first_token_ms: 312.50
stream_chunks: 47
error: none
```

### Error Codes

| Code                     | String           |
| ------------------------ | ---------------- |
| `LLM_OK`                 | `ok`             |
| `LLM_ERR_INVALID_PARAM`  | `invalid_param`  |
| `LLM_ERR_ALLOC`          | `alloc`          |
| `LLM_ERR_CURL`           | `curl`           |
| `LLM_ERR_HTTP`           | `http`           |
| `LLM_ERR_TIMEOUT`        | `timeout`        |
| `LLM_ERR_RATE_LIMIT`     | `rate_limit`     |
| `LLM_ERR_AUTH`           | `auth`           |
| `LLM_ERR_NOT_FOUND`      | `not_found`      |
| `LLM_ERR_SERVER`         | `server`         |
| `LLM_ERR_CONTEXT_LENGTH` | `context_length` |
| `LLM_ERR_CANCELLED`      | `cancelled`      |
| `LLM_ERR_THREAD`         | `thread`         |

### Retry Behaviour

Retries happen automatically with exponential backoff on:

- HTTP 429 (rate limit) — if `retry_on_rate_limit` is `true`
- HTTP 5xx server errors
- Request timeouts

Backoff: `delay = retry_delay_ms * backoff_multiplier^attempt`, capped at `retry_delay_max_ms`.

### CLI Tool (`llm.c`)

Build:

```sh
gcc -o llm llm.c -lcurl -lpthread -lm
```

Usage:

```
llm [OPTIONS] "prompt"

  -p, --provider <name>         openai, anthropic, groq, ollama, gemini, deepseek, vllm, ...
  -m, --model <name>            Model name
  -k, --api-key <key>           API key (or set <PROVIDER>_API_KEY env var)
  -u, --base-url <url>          Override base URL
  -s, --system <prompt>         System prompt
  -f, --file <path>             Prepend file contents as "<filename>:\n<content>\n"
  -t, --temperature <float>     Temperature
  -T, --max-tokens <int>        Max tokens
  -P, --top-p <float>           Top-p
  -K, --top-k <int>             Top-k
  -F, --frequency-penalty <f>   Frequency penalty
  -R, --presence-penalty <f>    Presence penalty
  -S, --stop <string>           Stop sequence (repeatable)
  -e, --seed <int>              Seed
  -j, --json                    JSON mode
  -r, --stream                  Stream output
  -n, --max-retries <int>       Max retries
  -d, --retry-delay <ms>        Initial retry delay
  -D, --retry-delay-max <ms>    Max retry delay
  -w, --timeout <seconds>       Timeout
  -x, --proxy <url>             HTTP proxy
  -o, --org <id>                OpenAI org ID
  -i, --project <id>            OpenAI project ID
  -H, --header <Key:Value>      Extra header (repeatable)
  -N, --no-retry-rate-limit     Disable rate limit retry
  -L, --no-verify-ssl           Disable SSL verification
  -v, --verbose                 Increase verbosity (-v, -vv, -vvv)
  -q, --stats                   Print stats to stderr after response
  -h, --help                    Show help
```

#### CLI Examples

```sh
# Simple query (API key from env)
export OPENAI_API_KEY=sk-...
llm "Explain recursion in one sentence"

# Anthropic with streaming
llm -p anthropic -m claude-opus-4-5 -r "Write a haiku about the sea"

# With a file attachment
llm -f main.c "Summarize what this code does"

# Multiple files (probably needs to match globs instead)
llm -f schema.sql -f queries.sql "Are there any N+1 query issues here?"

# System prompt + JSON mode
llm -s "You output JSON only." -j "List 3 planets with their diameters"

# Groq with stats
llm -p groq -m llama3-70b-8192 -q "What is the speed of light?"

# Local Ollama
llm -p ollama -m llama3 "Hello"

# Local vLLM
llm -p vllm -m meta-llama/Meta-Llama-3-8B-Instruct "Hello"

# DeepSeek
llm -p deepseek -m deepseek-chat "Explain transformers"

# Gemini via OpenAI-compatible endpoint
llm -p gemini -m gemini-2.0-flash "What is 2+2?"

# Custom endpoint
llm -p custom -u http://my-api.internal/v1 -m my-model "ping"

# Verbose debug output
llm -vv -p openai "test"

# Extra headers
llm -H "X-Request-ID: abc123" -H "X-Trace: xyz" "hello"
```

API keys are read from environment variables if `-k` is not provided:

| Provider     | Environment Variable |
| ------------ | -------------------- |
| OpenAI       | `OPENAI_API_KEY`     |
| Anthropic    | `ANTHROPIC_API_KEY`  |
| Groq         | `GROQ_API_KEY`       |
| Together     | `TOGETHER_API_KEY`   |
| Mistral      | `MISTRAL_API_KEY`    |
| Cohere       | `COHERE_API_KEY`     |
| Gemini       | `GEMINI_API_KEY`     |
| DeepSeek     | `DEEPSEEK_API_KEY`   |
| OpenRouter   | `OPENROUTER_API_KEY` |
| Perplexity   | `PERPLEXITY_API_KEY` |
| Fireworks    | `FIREWORKS_API_KEY`  |
| _(fallback)_ | `LLM_API_KEY`        |

### Thread Safety

`llm_complete` is thread-safe. Each call creates and destroys its own cURL handle. The `llm_client_t` and `llm_request_t` structs should not be mutated concurrently while a request is in flight. `llm_complete_batch` and `llm_complete_async` manage their own threads internally.

### License

Apache v2.0 License
