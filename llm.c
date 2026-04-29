#include "llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <getopt.h>

static void usage(const char* prog) {
  fprintf(
      stderr,
      "Usage: %s [OPTIONS] \"prompt\"\n"
      "\n"
      "Options:\n"
      "  -p, --provider <name>         Provider (openai, anthropic, groq, "
      "ollama, together,\n"
      "                                  mistral, cohere, gemini, deepseek, "
      "openrouter,\n"
      "                                  perplexity, fireworks, vllm, custom)\n"
      "  -m, --model <name>            Model name\n"
      "  -k, --api-key <key>           API key (or set <PROVIDER>_API_KEY env "
      "var)\n"
      "  -u, --base-url <url>          Override base URL\n"
      "  -s, --system <prompt>         System prompt\n"
      "  -f, --file <path>             Read file and prepend "
      "\"<filename>:\\n<content>\\n\" to prompt\n"
      "  -t, --temperature <float>     Temperature (default: 1.0)\n"
      "  -T, --max-tokens <int>        Max tokens (default: 4096)\n"
      "  -P, --top-p <float>           Top-p\n"
      "  -K, --top-k <int>             Top-k\n"
      "  -F, --frequency-penalty <f>   Frequency penalty\n"
      "  -R, --presence-penalty <f>    Presence penalty\n"
      "  -S, --stop <string>           Stop sequence (repeatable)\n"
      "  -e, --seed <int>              Seed for deterministic output\n"
      "  -j, --json                    Enable JSON mode\n"
      "  -r, --stream                  Stream output\n"
      "  -n, --max-retries <int>       Max retries (default: 3)\n"
      "  -d, --retry-delay <ms>        Initial retry delay in ms (default: "
      "1000)\n"
      "  -D, --retry-delay-max <ms>    Max retry delay in ms (default: 30000)\n"
      "  -w, --timeout <seconds>       Request timeout in seconds (default: "
      "30)\n"
      "  -x, --proxy <url>             HTTP proxy URL\n"
      "  -o, --org <id>                Organization ID (OpenAI)\n"
      "  -i, --project <id>            Project ID (OpenAI)\n"
      "  -H, --header <Key:Value>      Extra HTTP header (repeatable)\n"
      "  -N, --no-retry-rate-limit     Do not retry on 429 rate limit\n"
      "  -L, --no-verify-ssl           Disable SSL verification\n"
      "  -v, --verbose                 Increase verbosity (repeatable, up to "
      "-vvv)\n"
      "  -q, --stats                   Print request stats after response\n"
      "  -h, --help                    Show this help\n",
      prog);
}

static char* read_file(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "error: cannot open file: %s\n", path);
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    fclose(f);
    return NULL;
  }
  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t n = fread(buf, 1, (size_t)sz, f);
  buf[n] = '\0';
  fclose(f);
  return buf;
}

static const char* llm__getenv_api_key(llm_provider_t p) {
  const char* key = NULL;
  switch (p) {
  case LLM_PROVIDER_OPENAI:
    key = getenv("OPENAI_API_KEY");
    break;
  case LLM_PROVIDER_ANTHROPIC:
    key = getenv("ANTHROPIC_API_KEY");
    break;
  case LLM_PROVIDER_GROQ:
    key = getenv("GROQ_API_KEY");
    break;
  case LLM_PROVIDER_TOGETHER:
    key = getenv("TOGETHER_API_KEY");
    break;
  case LLM_PROVIDER_MISTRAL:
    key = getenv("MISTRAL_API_KEY");
    break;
  case LLM_PROVIDER_COHERE:
    key = getenv("COHERE_API_KEY");
    break;
  case LLM_PROVIDER_GEMINI:
    key = getenv("GEMINI_API_KEY");
    break;
  case LLM_PROVIDER_DEEPSEEK:
    key = getenv("DEEPSEEK_API_KEY");
    break;
  case LLM_PROVIDER_OPENROUTER:
    key = getenv("OPENROUTER_API_KEY");
    break;
  case LLM_PROVIDER_PERPLEXITY:
    key = getenv("PERPLEXITY_API_KEY");
    break;
  case LLM_PROVIDER_FIREWORKS:
    key = getenv("FIREWORKS_API_KEY");
    break;
  default:
    break;
  }
  if (!key)
    key = getenv("LLM_API_KEY");
  return key;
}

static void stream_cb(const char* delta, bool done, void* user_data) {
  (void)user_data;
  if (done) {
    fflush(stdout);
    return;
  }
  if (delta) {
    fputs(delta, stdout);
    fflush(stdout);
  }
}

#define MAX_STOP_SEQS 16
#define MAX_EXTRA_HEADERS 32
#define MAX_FILES 16

int main(int argc, char** argv) {
  llm_init();

  llm_client_t client;
  llm_client_init(&client);

  const char* provider_str = NULL;
  const char* model = NULL;
  const char* api_key = NULL;
  const char* base_url = NULL;
  const char* system = NULL;
  const char* files[MAX_FILES];
  int file_count = 0;
  double temperature = LLM_DEFAULT_TEMPERATURE;
  bool temperature_set = false;
  int max_tokens = LLM_DEFAULT_MAX_TOKENS;
  double top_p = 1.0;
  bool top_p_set = false;
  int top_k = 0;
  double freq_penalty = 0.0;
  double pres_penalty = 0.0;
  const char* stop_seqs[MAX_STOP_SEQS];
  int stop_count = 0;
  uint64_t seed = 0;
  bool use_seed = false;
  bool json_mode = false;
  bool stream = false;
  int max_retries = -1;
  int retry_delay = -1;
  int retry_delay_max = -1;
  long timeout = -1;
  const char* proxy = NULL;
  const char* org = NULL;
  const char* project = NULL;
  bool no_retry_rl = false;
  bool no_verify_ssl = false;
  int verbosity = 0;
  bool print_stats = false;
  const char* extra_headers[MAX_EXTRA_HEADERS];
  int extra_headers_count = 0;

  static struct option long_opts[] = {
      {"provider", required_argument, 0, 'p'},
      {"model", required_argument, 0, 'm'},
      {"api-key", required_argument, 0, 'k'},
      {"base-url", required_argument, 0, 'u'},
      {"system", required_argument, 0, 's'},
      {"file", required_argument, 0, 'f'},
      {"temperature", required_argument, 0, 't'},
      {"max-tokens", required_argument, 0, 'T'},
      {"top-p", required_argument, 0, 'P'},
      {"top-k", required_argument, 0, 'K'},
      {"frequency-penalty", required_argument, 0, 'F'},
      {"presence-penalty", required_argument, 0, 'R'},
      {"stop", required_argument, 0, 'S'},
      {"seed", required_argument, 0, 'e'},
      {"json", no_argument, 0, 'j'},
      {"stream", no_argument, 0, 'r'},
      {"max-retries", required_argument, 0, 'n'},
      {"retry-delay", required_argument, 0, 'd'},
      {"retry-delay-max", required_argument, 0, 'D'},
      {"timeout", required_argument, 0, 'w'},
      {"proxy", required_argument, 0, 'x'},
      {"org", required_argument, 0, 'o'},
      {"project", required_argument, 0, 'i'},
      {"header", required_argument, 0, 'H'},
      {"no-retry-rate-limit", no_argument, 0, 'N'},
      {"no-verify-ssl", no_argument, 0, 'L'},
      {"verbose", no_argument, 0, 'v'},
      {"stats", no_argument, 0, 'q'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int c;
  int opt_idx = 0;
  while ((c = getopt_long(argc, argv,
                          "p:m:k:u:s:f:t:T:P:K:F:R:S:e:jrn:d:D:w:x:o:i:H:NLvqh",
                          long_opts, &opt_idx)) != -1) {
    switch (c) {
    case 'p':
      provider_str = optarg;
      break;
    case 'm':
      model = optarg;
      break;
    case 'k':
      api_key = optarg;
      break;
    case 'u':
      base_url = optarg;
      break;
    case 's':
      system = optarg;
      break;
    case 'f':
      if (file_count < MAX_FILES)
        files[file_count++] = optarg;
      break;
    case 't':
      temperature = atof(optarg);
      temperature_set = true;
      break;
    case 'T':
      max_tokens = atoi(optarg);
      break;
    case 'P':
      top_p = atof(optarg);
      top_p_set = true;
      break;
    case 'K':
      top_k = atoi(optarg);
      break;
    case 'F':
      freq_penalty = atof(optarg);
      break;
    case 'R':
      pres_penalty = atof(optarg);
      break;
    case 'S':
      if (stop_count < MAX_STOP_SEQS)
        stop_seqs[stop_count++] = optarg;
      break;
    case 'e':
      seed = (uint64_t)strtoull(optarg, NULL, 10);
      use_seed = true;
      break;
    case 'j':
      json_mode = true;
      break;
    case 'r':
      stream = true;
      break;
    case 'n':
      max_retries = atoi(optarg);
      break;
    case 'd':
      retry_delay = atoi(optarg);
      break;
    case 'D':
      retry_delay_max = atoi(optarg);
      break;
    case 'w':
      timeout = atol(optarg);
      break;
    case 'x':
      proxy = optarg;
      break;
    case 'o':
      org = optarg;
      break;
    case 'i':
      project = optarg;
      break;
    case 'H':
      if (extra_headers_count < MAX_EXTRA_HEADERS)
        extra_headers[extra_headers_count++] = optarg;
      break;
    case 'N':
      no_retry_rl = true;
      break;
    case 'L':
      no_verify_ssl = true;
      break;
    case 'v':
      verbosity++;
      break;
    case 'q':
      print_stats = true;
      break;
    case 'h':
      usage(argv[0]);
      llm_cleanup();
      return 0;
    default:
      usage(argv[0]);
      llm_cleanup();
      return 1;
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "error: no prompt provided\n\n");
    usage(argv[0]);
    llm_cleanup();
    return 1;
  }

  const char* raw_prompt = argv[optind];

  llm__buf_t prompt_buf = llm__buf_new();
  for (int i = 0; i < file_count; i++) {
    const char* fpath = files[i];
    char* fcontent = read_file(fpath);
    if (!fcontent) {
      llm__buf_free(&prompt_buf);
      llm_cleanup();
      return 1;
    }
    const char* fname = strrchr(fpath, '/');
    fname = fname ? fname + 1 : fpath;
    llm__buf_append(&prompt_buf, fname, strlen(fname));
    llm__buf_append(&prompt_buf, ":\n", 2);
    llm__buf_append(&prompt_buf, fcontent, strlen(fcontent));
    llm__buf_append(&prompt_buf, "\n", 1);
    free(fcontent);
  }
  llm__buf_append(&prompt_buf, raw_prompt, strlen(raw_prompt));

  client.provider = provider_str ? llm_provider_from_string(provider_str)
                                 : LLM_PROVIDER_OPENAI;
  client.model = model ? llm__strdup(model) : NULL;
  client.base_url = base_url ? llm__strdup(base_url) : NULL;
  client.verbosity = verbosity;

  if (api_key) {
    client.api_key = llm__strdup(api_key);
  } else {
    const char* env_key = llm__getenv_api_key(client.provider);
    if (env_key)
      client.api_key = llm__strdup(env_key);
  }

  if (max_retries >= 0)
    client.max_retries = max_retries;
  if (retry_delay >= 0)
    client.retry_delay_ms = retry_delay;
  if (retry_delay_max >= 0)
    client.retry_delay_max_ms = retry_delay_max;
  if (timeout > 0)
    client.timeout_seconds = timeout;
  if (proxy)
    client.proxy = llm__strdup(proxy);
  if (org)
    client.org_id = llm__strdup(org);
  if (project)
    client.project_id = llm__strdup(project);
  if (no_retry_rl)
    client.retry_on_rate_limit = false;
  if (no_verify_ssl)
    client.verify_ssl = false;

  for (int i = 0; i < extra_headers_count; i++) {
    const char* hdr = extra_headers[i];
    const char* colon = strchr(hdr, ':');
    if (!colon) {
      fprintf(stderr, "warning: ignoring malformed header (no colon): %s\n",
              hdr);
      continue;
    }
    char* hkey = llm__strndup(hdr, (size_t)(colon - hdr));
    const char* hval = colon + 1;
    while (*hval == ' ')
      hval++;
    llm_client_add_header(&client, hkey, hval);
    free(hkey);
  }

  llm_request_t req;
  llm_request_init(&req);

  llm_message_t msg = llm_msg_user(prompt_buf.buffer);
  llm__buf_free(&prompt_buf);

  req.messages = &msg;
  req.messages_count = 1;
  req.max_tokens = max_tokens;
  req.top_k = top_k;
  req.frequency_penalty = freq_penalty;
  req.presence_penalty = pres_penalty;
  req.json_mode = json_mode;
  req.stream = stream;
  req.use_seed = use_seed;
  req.seed = seed;
  req.system = system ? llm__strdup(system) : NULL;

  if (temperature_set)
    req.temperature = temperature;
  if (top_p_set)
    req.top_p = top_p;

  if (stop_count > 0) {
    req.stop = (char**)stop_seqs;
    req.stop_count = stop_count;
  }

  if (stream)
    req.stream_callback = stream_cb;

  llm_response_t* resp = llm_complete(&client, &req);

  free(msg.content);
  free(req.system);

  if (!resp) {
    fprintf(stderr, "error: null response\n");
    llm_client_free(&client);
    llm_cleanup();
    return 1;
  }

  if (resp->error != LLM_OK) {
    fprintf(stderr, "error [%s]: %s\n", llm_error_string(resp->error),
            resp->error_message ? resp->error_message : "(no message)");
    if (print_stats)
      llm_print_stats(resp, stderr);
    llm_response_free(resp);
    free(resp);
    llm_client_free(&client);
    llm_cleanup();
    return 1;
  }

  if (!stream) {
    if (resp->content)
      puts(resp->content);
  } else {
    puts("");
  }

  if (print_stats)
    llm_print_stats(resp, stderr);

  llm_response_free(resp);
  free(resp);
  llm_client_free(&client);
  llm_cleanup();
  return 0;
}
