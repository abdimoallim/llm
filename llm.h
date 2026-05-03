// Copyright 2026 Abdi Moalim
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LLM_H
#define LLM_H

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <curl/curl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLM_VERSION_MAJOR 1
#define LLM_VERSION_MINOR 0
#define LLM_VERSION_PATCH 0

#define LLM_MAX_MESSAGES 256
#define LLM_MAX_TOOLS 64
#define LLM_MAX_BATCH 64
#define LLM_DEFAULT_TIMEOUT 30L
#define LLM_DEFAULT_MAX_RETRIES 3
#define LLM_DEFAULT_RETRY_DELAY_MS 1000
#define LLM_DEFAULT_MAX_TOKENS 4096
#define LLM_DEFAULT_TEMPERATURE 1.0

typedef enum {
  LLM_OK = 0,
  LLM_ERR_INVALID_PARAM,
  LLM_ERR_ALLOC,
  LLM_ERR_CURL,
  LLM_ERR_HTTP,
  LLM_ERR_PARSE,
  LLM_ERR_TIMEOUT,
  LLM_ERR_RATE_LIMIT,
  LLM_ERR_AUTH,
  LLM_ERR_NOT_FOUND,
  LLM_ERR_SERVER,
  LLM_ERR_UNSUPPORTED,
  LLM_ERR_CANCELLED,
  LLM_ERR_THREAD,
  LLM_ERR_CONTEXT_LENGTH,
} llm_error_t;

typedef enum {
  LLM_PROVIDER_OPENAI,
  LLM_PROVIDER_ANTHROPIC,
  LLM_PROVIDER_GROQ,
  LLM_PROVIDER_OLLAMA,
  LLM_PROVIDER_TOGETHER,
  LLM_PROVIDER_MISTRAL,
  LLM_PROVIDER_COHERE,
  LLM_PROVIDER_GEMINI,
  LLM_PROVIDER_DEEPSEEK,
  LLM_PROVIDER_OPENROUTER,
  LLM_PROVIDER_PERPLEXITY,
  LLM_PROVIDER_FIREWORKS,
  LLM_PROVIDER_VLLM,
  LLM_PROVIDER_CUSTOM,
} llm_provider_t;

typedef enum {
  LLM_ROLE_SYSTEM,
  LLM_ROLE_USER,
  LLM_ROLE_ASSISTANT,
  LLM_ROLE_TOOL,
} llm_role_t;

typedef enum {
  LLM_FINISH_STOP,
  LLM_FINISH_LENGTH,
  LLM_FINISH_TOOL_CALL,
  LLM_FINISH_CONTENT_FILTER,
  LLM_FINISH_ERROR,
  LLM_FINISH_UNKNOWN,
} llm_finish_reason_t;

typedef enum {
  LLM_TOOL_TYPE_FUNCTION,
} llm_tool_type_t;

typedef struct {
  char* key;
  char* value;
} llm_header_t;

typedef struct {
  char* name;
  char* description;
  char* parameters_json;
} llm_function_def_t;

typedef struct {
  llm_tool_type_t type;
  llm_function_def_t function;
} llm_tool_def_t;

typedef struct {
  char* id;
  char* name;
  char* arguments_json;
} llm_tool_call_t;

typedef struct {
  llm_role_t role;
  char* content;
  char* name;
  char* tool_call_id;
  llm_tool_call_t* tool_calls;
  int tool_calls_count;
} llm_message_t;

typedef struct {
  int prompt_tokens;
  int completion_tokens;
  int total_tokens;
  bool provided;
} llm_usage_t;

typedef struct {
  double latency_ms;
  double time_to_first_token_ms;
  int retries;
  int stream_chunks;
  bool has_time_to_first_token;
} llm_stats_t;

typedef struct {
  char* id;
  char* model;
  char* content;
  llm_tool_call_t* tool_calls;
  int tool_calls_count;
  llm_finish_reason_t finish_reason;
  llm_usage_t usage;
  llm_stats_t stats;
  llm_error_t error;
  char* error_message;
  int http_status;
  void* user_data;
} llm_response_t;

typedef void (*llm_stream_callback_t)(const char* delta, bool done,
                                      void* user_data);
typedef void (*llm_async_callback_t)(llm_response_t* response, void* user_data);

typedef struct {
  llm_provider_t provider;
  char* api_key;
  char* base_url;
  char* model;
  long timeout_seconds;
  int max_retries;
  int retry_delay_ms;
  int retry_delay_max_ms;
  double retry_backoff_multiplier;
  bool retry_on_rate_limit;
  llm_header_t* extra_headers;
  int extra_headers_count;
  int verbosity;
  bool verify_ssl;
  char* proxy;
  char* org_id;
  char* project_id;
} llm_client_t;

typedef struct {
  llm_message_t* messages;
  int messages_count;
  char* model;
  int max_tokens;
  double temperature;
  double top_p;
  double frequency_penalty;
  double presence_penalty;
  int top_k;
  char** stop;
  int stop_count;
  bool stream;
  llm_stream_callback_t stream_callback;
  void* stream_user_data;
  llm_tool_def_t* tools;
  int tools_count;
  char* tool_choice;
  char* system;
  double min_p;
  bool json_mode;
  char* json_schema;
  uint64_t seed;
  bool use_seed;
  void* user_data;
} llm_request_t;

static void llm__log(int verbosity, int level, const char* fmt, ...) {
  if (verbosity < level)
    return;
  va_list args;
  va_start(args, fmt);
  const char* prefix = level == 1   ? "[LLM INFO] "
                       : level == 2 ? "[LLM DEBUG] "
                                    : "[LLM TRACE] ";
  fprintf(stderr, "%s", prefix);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
  va_end(args);
}

#define LLM__LOG_INFO(c, ...) llm__log((c)->verbosity, 1, __VA_ARGS__)
#define LLM__LOG_DEBUG(c, ...) llm__log((c)->verbosity, 2, __VA_ARGS__)
#define LLM__LOG_TRACE(c, ...) llm__log((c)->verbosity, 3, __VA_ARGS__)

static char* llm__strdup(const char* s) {
  if (!s)
    return NULL;
  size_t len = strlen(s) + 1;
  char* out = (char*)malloc(len);
  if (out)
    memcpy(out, s, len);
  return out;
}

static char* llm__strndup(const char* s, size_t n) {
  if (!s)
    return NULL;
  char* out = (char*)malloc(n + 1);
  if (out) {
    memcpy(out, s, n);
    out[n] = '\0';
  }
  return out;
}

typedef struct {
  char* buffer;
  size_t size;
  size_t capacity;
} llm__buf_t;

static llm__buf_t llm__buf_new(void) {
  llm__buf_t b = {0};
  b.capacity = 4096;
  b.buffer = (char*)malloc(b.capacity);
  if (b.buffer)
    b.buffer[0] = '\0';
  return b;
}

static void llm__buf_free(llm__buf_t* b) {
  free(b->buffer);
  b->buffer = NULL;
  b->size = 0;
  b->capacity = 0;
}

static bool llm__buf_append(llm__buf_t* b, const char* data, size_t len) {
  if (!b->buffer)
    return false;
  while (b->size + len + 1 > b->capacity) {
    size_t nc = b->capacity * 2;
    char* nb = (char*)realloc(b->buffer, nc);
    if (!nb)
      return false;
    b->buffer = nb;
    b->capacity = nc;
  }
  memcpy(b->buffer + b->size, data, len);
  b->size += len;
  b->buffer[b->size] = '\0';
  return true;
}

static size_t llm__curl_write(void* data, size_t size, size_t nmemb,
                              void* userp) {
  size_t real = size * nmemb;
  llm__buf_append((llm__buf_t*)userp, (char*)data, real);
  return real;
}

static char* llm__json_escape(const char* s) {
  if (!s)
    return llm__strdup("null");
  size_t len = strlen(s);
  size_t cap = len * 2 + 3;
  char* out = (char*)malloc(cap);
  if (!out)
    return NULL;
  size_t j = 0;
  out[j++] = '"';
  for (size_t i = 0; i < len; i++) {
    unsigned char c = (unsigned char)s[i];
    if (j + 8 >= cap) {
      cap = cap * 2 + 16;
      char* tmp = (char*)realloc(out, cap);
      if (!tmp) {
        free(out);
        return NULL;
      }
      out = tmp;
    }
    if (c == '"') {
      out[j++] = '\\';
      out[j++] = '"';
    } else if (c == '\\') {
      out[j++] = '\\';
      out[j++] = '\\';
    } else if (c == '\n') {
      out[j++] = '\\';
      out[j++] = 'n';
    } else if (c == '\r') {
      out[j++] = '\\';
      out[j++] = 'r';
    } else if (c == '\t') {
      out[j++] = '\\';
      out[j++] = 't';
    } else if (c < 0x20) {
      j += snprintf(out + j, cap - j, "\\u%04x", c);
    } else {
      out[j++] = (char)c;
    }
  }
  out[j++] = '"';
  out[j] = '\0';
  return out;
}

static const char* llm__json_find_key(const char* json, const char* key) {
  if (!json || !key)
    return NULL;
  size_t klen = strlen(key);
  char needle[256];
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  size_t nlen = klen + 2;
  const char* p = json;
  int depth = 0;
  bool in_str = false;
  while (*p) {
    if (in_str) {
      if (*p == '\\') {
        p++;
        if (*p)
          p++;
        continue;
      }
      if (*p == '"')
        in_str = false;
      p++;
      continue;
    }
    if (*p == '"') {
      if (depth == 1 && strncmp(p, needle, nlen) == 0) {
        const char* after = p + nlen;
        while (*after == ' ' || *after == '\t' || *after == '\n' ||
               *after == '\r')
          after++;
        if (*after == ':')
          return after + 1;
      }
      in_str = true;
      p++;
      continue;
    }
    if (*p == '{' || *p == '[')
      depth++;
    else if (*p == '}' || *p == ']')
      depth--;
    p++;
  }
  return NULL;
}

static char* llm__json_get_string(const char* json, const char* key) {
  const char* val = llm__json_find_key(json, key);
  if (!val)
    return NULL;
  while (*val == ' ' || *val == '\t' || *val == '\n' || *val == '\r')
    val++;
  if (*val != '"')
    return NULL;
  val++;
  size_t cap = 256;
  char* out = (char*)malloc(cap);
  if (!out)
    return NULL;
  size_t j = 0;
  while (*val && *val != '"') {
    if (j + 8 >= cap) {
      cap *= 2;
      char* tmp = (char*)realloc(out, cap);
      if (!tmp) {
        free(out);
        return NULL;
      }
      out = tmp;
    }
    if (*val == '\\') {
      val++;
      switch (*val) {
      case '"':
        out[j++] = '"';
        break;
      case '\\':
        out[j++] = '\\';
        break;
      case '/':
        out[j++] = '/';
        break;
      case 'n':
        out[j++] = '\n';
        break;
      case 'r':
        out[j++] = '\r';
        break;
      case 't':
        out[j++] = '\t';
        break;
      case 'b':
        out[j++] = '\b';
        break;
      case 'f':
        out[j++] = '\f';
        break;
      case 'u': {
        unsigned int cp = 0;
        sscanf(val + 1, "%4x", &cp);
        val += 4;
        if (cp < 0x80) {
          out[j++] = (char)cp;
        } else if (cp < 0x800) {
          out[j++] = (char)(0xC0 | (cp >> 6));
          out[j++] = (char)(0x80 | (cp & 0x3F));
        } else {
          out[j++] = (char)(0xE0 | (cp >> 12));
          out[j++] = (char)(0x80 | ((cp >> 6) & 0x3F));
          out[j++] = (char)(0x80 | (cp & 0x3F));
        }
        break;
      }
      default:
        out[j++] = *val;
        break;
      }
    } else {
      out[j++] = *val;
    }
    val++;
  }
  out[j] = '\0';
  return out;
}

static long long llm__json_get_int(const char* json, const char* key) {
  const char* val = llm__json_find_key(json, key);
  if (!val)
    return 0;
  while (*val == ' ' || *val == '\t' || *val == '\n' || *val == '\r')
    val++;
  return strtoll(val, NULL, 10);
}

static char* llm__json_get_nested(const char* json, const char* key) {
  const char* val = llm__json_find_key(json, key);
  if (!val)
    return NULL;
  while (*val == ' ' || *val == '\t' || *val == '\n' || *val == '\r')
    val++;
  if (*val != '{' && *val != '[')
    return NULL;
  char open = *val, close = (open == '{') ? '}' : ']';
  int depth = 0;
  bool in_str = false;
  const char* start = val;
  while (*val) {
    if (in_str) {
      if (*val == '\\') {
        val++;
        if (*val)
          val++;
        continue;
      }
      if (*val == '"')
        in_str = false;
      val++;
      continue;
    }
    if (*val == '"') {
      in_str = true;
      val++;
      continue;
    }
    if (*val == open)
      depth++;
    else if (*val == close) {
      depth--;
      if (depth == 0)
        return llm__strndup(start, (size_t)(val - start + 1));
    }
    val++;
  }
  return NULL;
}

static char* llm__json_array_get(const char* arr, int index) {
  if (!arr || *arr != '[')
    return NULL;
  const char* p = arr + 1;
  int i = 0;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
      p++;
    if (*p == ']')
      break;
    const char* start = p;
    int depth = 0;
    bool in_str = false;
    while (*p) {
      if (in_str) {
        if (*p == '\\') {
          p++;
          if (*p)
            p++;
          continue;
        }
        if (*p == '"')
          in_str = false;
        p++;
        continue;
      }
      if (*p == '"') {
        in_str = true;
        p++;
        continue;
      }
      if (*p == '{' || *p == '[')
        depth++;
      else if (*p == '}' || *p == ']') {
        if (depth == 0)
          break;
        depth--;
      } else if (*p == ',' && depth == 0)
        break;
      p++;
    }
    if (i == index)
      return llm__strndup(start, (size_t)(p - start));
    i++;
    if (*p == ',')
      p++;
  }
  return NULL;
}

static int llm__json_array_len(const char* arr) {
  if (!arr || *arr != '[')
    return 0;
  const char* p = arr + 1;
  int count = 0;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
      p++;
    if (*p == ']')
      break;
    int depth = 0;
    bool in_str = false;
    bool adv = false;
    while (*p) {
      if (in_str) {
        if (*p == '\\') {
          p++;
          if (*p)
            p++;
          continue;
        }
        if (*p == '"')
          in_str = false;
        p++;
        continue;
      }
      if (*p == '"') {
        in_str = true;
        p++;
        continue;
      }
      if (*p == '{' || *p == '[')
        depth++;
      else if (*p == '}' || *p == ']') {
        if (depth == 0) {
          adv = true;
          break;
        }
        depth--;
      } else if (*p == ',' && depth == 0) {
        adv = true;
        break;
      }
      p++;
    }
    count++;
    if (!adv)
      break;
    if (*p == ',')
      p++;
    else
      break;
  }
  return count;
}

static const char* llm__provider_default_url(llm_provider_t p) {
  switch (p) {
  case LLM_PROVIDER_OPENAI:
    return "https://api.openai.com/v1";
  case LLM_PROVIDER_ANTHROPIC:
    return "https://api.anthropic.com/v1";
  case LLM_PROVIDER_GROQ:
    return "https://api.groq.com/openai/v1";
  case LLM_PROVIDER_OLLAMA:
    return "http://localhost:11434/v1";
  case LLM_PROVIDER_TOGETHER:
    return "https://api.together.xyz/v1";
  case LLM_PROVIDER_MISTRAL:
    return "https://api.mistral.ai/v1";
  case LLM_PROVIDER_COHERE:
    return "https://api.cohere.ai/v1";
  case LLM_PROVIDER_GEMINI:
    return "https://generativelanguage.googleapis.com/v1beta/openai";
  case LLM_PROVIDER_DEEPSEEK:
    return "https://api.deepseek.com/v1";
  case LLM_PROVIDER_OPENROUTER:
    return "https://openrouter.ai/api/v1";
  case LLM_PROVIDER_PERPLEXITY:
    return "https://api.perplexity.ai";
  case LLM_PROVIDER_FIREWORKS:
    return "https://api.fireworks.ai/inference/v1";
  case LLM_PROVIDER_VLLM:
    return "http://localhost:8000/v1";
  default:
    return NULL;
  }
}

static bool llm__provider_uses_openai_compat(llm_provider_t p) {
  return (p != LLM_PROVIDER_ANTHROPIC && p != LLM_PROVIDER_COHERE);
}

static void llm_client_init(llm_client_t* c) {
  memset(c, 0, sizeof(*c));
  c->timeout_seconds = LLM_DEFAULT_TIMEOUT;
  c->max_retries = LLM_DEFAULT_MAX_RETRIES;
  c->retry_delay_ms = LLM_DEFAULT_RETRY_DELAY_MS;
  c->retry_delay_max_ms = 30000;
  c->retry_backoff_multiplier = 2.0;
  c->retry_on_rate_limit = true;
  c->verbosity = 0;
  c->verify_ssl = true;
}

static void llm_client_free(llm_client_t* c) {
  free(c->api_key);
  free(c->base_url);
  free(c->model);
  free(c->proxy);
  free(c->org_id);
  free(c->project_id);
  for (int i = 0; i < c->extra_headers_count; i++) {
    free(c->extra_headers[i].key);
    free(c->extra_headers[i].value);
  }
  free(c->extra_headers);
  memset(c, 0, sizeof(*c));
}

static void llm_request_init(llm_request_t* r) {
  memset(r, 0, sizeof(*r));
  r->max_tokens = LLM_DEFAULT_MAX_TOKENS;
  r->temperature = LLM_DEFAULT_TEMPERATURE;
  r->top_p = 1.0;
}

static void llm_response_free(llm_response_t* r) {
  if (!r)
    return;
  free(r->id);
  free(r->model);
  free(r->content);
  free(r->error_message);
  for (int i = 0; i < r->tool_calls_count; i++) {
    free(r->tool_calls[i].id);
    free(r->tool_calls[i].name);
    free(r->tool_calls[i].arguments_json);
  }
  free(r->tool_calls);
  memset(r, 0, sizeof(*r));
}

static void llm_message_free(llm_message_t* m) {
  if (!m)
    return;
  free(m->content);
  free(m->name);
  free(m->tool_call_id);
  for (int i = 0; i < m->tool_calls_count; i++) {
    free(m->tool_calls[i].id);
    free(m->tool_calls[i].name);
    free(m->tool_calls[i].arguments_json);
  }
  free(m->tool_calls);
}

static char* llm__build_openai_request(llm_client_t* c, llm_request_t* r) {
  llm__buf_t b = llm__buf_new();
  const char* model = r->model ? r->model : (c->model ? c->model : "gpt-4o");
  char* me = llm__json_escape(model);
  llm__buf_append(&b, "{\"model\":", 9);
  llm__buf_append(&b, me, strlen(me));
  free(me);

  char tmp[256];
  snprintf(tmp, sizeof(tmp), ",\"max_tokens\":%d,\"temperature\":%.6g",
           r->max_tokens, r->temperature);
  llm__buf_append(&b, tmp, strlen(tmp));
  if (r->top_p != 1.0) {
    snprintf(tmp, sizeof(tmp), ",\"top_p\":%.6g", r->top_p);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->frequency_penalty != 0.0) {
    snprintf(tmp, sizeof(tmp), ",\"frequency_penalty\":%.6g",
             r->frequency_penalty);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->presence_penalty != 0.0) {
    snprintf(tmp, sizeof(tmp), ",\"presence_penalty\":%.6g",
             r->presence_penalty);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->use_seed) {
    snprintf(tmp, sizeof(tmp), ",\"seed\":%llu", (unsigned long long)r->seed);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->stream) {
    llm__buf_append(&b, ",\"stream\":true", 14);
  }
  if (r->json_mode)
    llm__buf_append(&b, ",\"response_format\":{\"type\":\"json_object\"}", 41);
  if (r->json_schema) {
    llm__buf_append(
        &b,
        ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":", 53);
    llm__buf_append(&b, r->json_schema, strlen(r->json_schema));
    llm__buf_append(&b, "}", 1);
  }
  if (r->stop_count > 0) {
    llm__buf_append(&b, ",\"stop\":[", 9);
    for (int i = 0; i < r->stop_count; i++) {
      if (i)
        llm__buf_append(&b, ",", 1);
      char* s = llm__json_escape(r->stop[i]);
      llm__buf_append(&b, s, strlen(s));
      free(s);
    }
    llm__buf_append(&b, "]", 1);
  }
  if (r->tools_count > 0) {
    llm__buf_append(&b, ",\"tools\":[", 10);
    for (int i = 0; i < r->tools_count; i++) {
      if (i)
        llm__buf_append(&b, ",", 1);
      char* fn = llm__json_escape(r->tools[i].function.name);
      char* fd = llm__json_escape(r->tools[i].function.description);
      llm__buf_append(&b, "{\"type\":\"function\",\"function\":{\"name\":", 38);
      llm__buf_append(&b, fn, strlen(fn));
      llm__buf_append(&b, ",\"description\":", 15);
      llm__buf_append(&b, fd, strlen(fd));
      free(fn);
      free(fd);
      if (r->tools[i].function.parameters_json) {
        llm__buf_append(&b, ",\"parameters\":", 14);
        llm__buf_append(&b, r->tools[i].function.parameters_json,
                        strlen(r->tools[i].function.parameters_json));
      }
      llm__buf_append(&b, "}}", 2);
    }
    llm__buf_append(&b, "]", 1);
    if (r->tool_choice) {
      char* tc = llm__json_escape(r->tool_choice);
      llm__buf_append(&b, ",\"tool_choice\":", 15);
      llm__buf_append(&b, tc, strlen(tc));
      free(tc);
    }
  }

  llm__buf_append(&b, ",\"messages\":[", 13);
  bool first = true;
  if (r->system) {
    char* sc = llm__json_escape(r->system);
    llm__buf_append(&b, "{\"role\":\"system\",\"content\":", 27);
    llm__buf_append(&b, sc, strlen(sc));
    llm__buf_append(&b, "}", 1);
    free(sc);
    first = false;
  }
  for (int i = 0; i < r->messages_count; i++) {
    llm_message_t* m = &r->messages[i];
    if (!first)
      llm__buf_append(&b, ",", 1);
    first = false;
    const char* rs;
    switch (m->role) {
    case LLM_ROLE_SYSTEM:
      rs = "system";
      break;
    case LLM_ROLE_ASSISTANT:
      rs = "assistant";
      break;
    case LLM_ROLE_TOOL:
      rs = "tool";
      break;
    default:
      rs = "user";
      break;
    }
    llm__buf_append(&b, "{\"role\":\"", 9);
    llm__buf_append(&b, rs, strlen(rs));
    llm__buf_append(&b, "\"", 1);
    if (m->role == LLM_ROLE_TOOL) {
      char* tid = llm__json_escape(m->tool_call_id ? m->tool_call_id : "");
      llm__buf_append(&b, ",\"tool_call_id\":", 16);
      llm__buf_append(&b, tid, strlen(tid));
      free(tid);
    }
    if (m->content) {
      char* ct = llm__json_escape(m->content);
      llm__buf_append(&b, ",\"content\":", 11);
      llm__buf_append(&b, ct, strlen(ct));
      free(ct);
    } else if (m->tool_calls_count == 0)
      llm__buf_append(&b, ",\"content\":null", 15);
    if (m->tool_calls_count > 0) {
      llm__buf_append(&b, ",\"tool_calls\":[", 15);
      for (int j = 0; j < m->tool_calls_count; j++) {
        if (j)
          llm__buf_append(&b, ",", 1);
        char* tid = llm__json_escape(m->tool_calls[j].id);
        char* tn = llm__json_escape(m->tool_calls[j].name);
        const char* ta = m->tool_calls[j].arguments_json
                             ? m->tool_calls[j].arguments_json
                             : "{}";
        char* tae = llm__json_escape(ta);
        llm__buf_append(&b, "{\"id\":", 6);
        llm__buf_append(&b, tid, strlen(tid));
        llm__buf_append(&b,
                        ",\"type\":\"function\",\"function\":{\"name\":", 38);
        llm__buf_append(&b, tn, strlen(tn));
        llm__buf_append(&b, ",\"arguments\":", 13);
        llm__buf_append(&b, tae, strlen(tae));
        llm__buf_append(&b, "}}", 2);
        free(tid);
        free(tn);
        free(tae);
      }
      llm__buf_append(&b, "]", 1);
    }
    llm__buf_append(&b, "}", 1);
  }
  llm__buf_append(&b, "]}", 2);
  char* result = llm__strdup(b.buffer);
  llm__buf_free(&b);
  return result;
}

static char* llm__build_anthropic_request(llm_client_t* c, llm_request_t* r) {
  llm__buf_t b = llm__buf_new();
  const char* model =
      r->model ? r->model : (c->model ? c->model : "claude-opus-4-5");
  char* me = llm__json_escape(model);
  llm__buf_append(&b, "{\"model\":", 9);
  llm__buf_append(&b, me, strlen(me));
  free(me);
  char tmp[256];
  snprintf(tmp, sizeof(tmp), ",\"max_tokens\":%d", r->max_tokens);
  llm__buf_append(&b, tmp, strlen(tmp));
  if (r->temperature != LLM_DEFAULT_TEMPERATURE) {
    snprintf(tmp, sizeof(tmp), ",\"temperature\":%.6g", r->temperature);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->top_p != 1.0) {
    snprintf(tmp, sizeof(tmp), ",\"top_p\":%.6g", r->top_p);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->top_k > 0) {
    snprintf(tmp, sizeof(tmp), ",\"top_k\":%d", r->top_k);
    llm__buf_append(&b, tmp, strlen(tmp));
  }
  if (r->stream)
    llm__buf_append(&b, ",\"stream\":true", 14);
  if (r->stop_count > 0) {
    llm__buf_append(&b, ",\"stop_sequences\":[", 19);
    for (int i = 0; i < r->stop_count; i++) {
      if (i)
        llm__buf_append(&b, ",", 1);
      char* s = llm__json_escape(r->stop[i]);
      llm__buf_append(&b, s, strlen(s));
      free(s);
    }
    llm__buf_append(&b, "]", 1);
  }
  if (r->system) {
    char* sc = llm__json_escape(r->system);
    llm__buf_append(&b, ",\"system\":", 10);
    llm__buf_append(&b, sc, strlen(sc));
    free(sc);
  }
  if (r->tools_count > 0) {
    llm__buf_append(&b, ",\"tools\":[", 10);
    for (int i = 0; i < r->tools_count; i++) {
      if (i)
        llm__buf_append(&b, ",", 1);
      char* fn = llm__json_escape(r->tools[i].function.name);
      char* fd = llm__json_escape(r->tools[i].function.description);
      llm__buf_append(&b, "{\"name\":", 8);
      llm__buf_append(&b, fn, strlen(fn));
      llm__buf_append(&b, ",\"description\":", 15);
      llm__buf_append(&b, fd, strlen(fd));
      free(fn);
      free(fd);
      if (r->tools[i].function.parameters_json) {
        llm__buf_append(&b, ",\"input_schema\":", 16);
        llm__buf_append(&b, r->tools[i].function.parameters_json,
                        strlen(r->tools[i].function.parameters_json));
      }
      llm__buf_append(&b, "}", 1);
    }
    llm__buf_append(&b, "]", 1);
  }
  llm__buf_append(&b, ",\"messages\":[", 13);
  bool first = true;
  for (int i = 0; i < r->messages_count; i++) {
    llm_message_t* m = &r->messages[i];
    if (m->role == LLM_ROLE_SYSTEM)
      continue;
    if (!first)
      llm__buf_append(&b, ",", 1);
    first = false;
    const char* rs = (m->role == LLM_ROLE_ASSISTANT) ? "assistant" : "user";
    llm__buf_append(&b, "{\"role\":\"", 9);
    llm__buf_append(&b, rs, strlen(rs));
    llm__buf_append(&b, "\",\"content\":", 12);
    if (m->tool_calls_count > 0) {
      llm__buf_append(&b, "[", 1);
      bool fc = true;
      if (m->content) {
        char* ct = llm__json_escape(m->content);
        llm__buf_append(&b, "{\"type\":\"text\",\"text\":", 22);
        llm__buf_append(&b, ct, strlen(ct));
        llm__buf_append(&b, "}", 1);
        free(ct);
        fc = false;
      }
      for (int j = 0; j < m->tool_calls_count; j++) {
        if (!fc)
          llm__buf_append(&b, ",", 1);
        fc = false;
        char* tid = llm__json_escape(m->tool_calls[j].id);
        char* tn = llm__json_escape(m->tool_calls[j].name);
        const char* inp = m->tool_calls[j].arguments_json
                              ? m->tool_calls[j].arguments_json
                              : "{}";
        llm__buf_append(&b, "{\"type\":\"tool_use\",\"id\":", 24);
        llm__buf_append(&b, tid, strlen(tid));
        llm__buf_append(&b, ",\"name\":", 8);
        llm__buf_append(&b, tn, strlen(tn));
        llm__buf_append(&b, ",\"input\":", 9);
        llm__buf_append(&b, inp, strlen(inp));
        llm__buf_append(&b, "}", 1);
        free(tid);
        free(tn);
      }
      llm__buf_append(&b, "]", 1);
    } else if (m->role == LLM_ROLE_TOOL) {
      char* tid = llm__json_escape(m->tool_call_id ? m->tool_call_id : "");
      char* ct = llm__json_escape(m->content ? m->content : "");
      llm__buf_append(&b, "[{\"type\":\"tool_result\",\"tool_use_id\":", 37);
      llm__buf_append(&b, tid, strlen(tid));
      llm__buf_append(&b, ",\"content\":", 11);
      llm__buf_append(&b, ct, strlen(ct));
      llm__buf_append(&b, "}]", 2);
      free(tid);
      free(ct);
    } else {
      char* ct = llm__json_escape(m->content ? m->content : "");
      llm__buf_append(&b, ct, strlen(ct));
      free(ct);
    }
    llm__buf_append(&b, "}", 1);
  }
  llm__buf_append(&b, "]}", 2);
  char* result = llm__strdup(b.buffer);
  llm__buf_free(&b);
  return result;
}

static char* llm__build_cohere_request(llm_client_t* c, llm_request_t* r) {
  llm__buf_t b = llm__buf_new();
  const char* model =
      r->model ? r->model : (c->model ? c->model : "command-r-plus");
  char* me = llm__json_escape(model);
  llm__buf_append(&b, "{\"model\":", 9);
  llm__buf_append(&b, me, strlen(me));
  free(me);
  char tmp[256];
  snprintf(tmp, sizeof(tmp), ",\"max_tokens\":%d,\"temperature\":%.6g",
           r->max_tokens, r->temperature);
  llm__buf_append(&b, tmp, strlen(tmp));
  if (r->stream)
    llm__buf_append(&b, ",\"stream\":true", 14);
  if (r->system) {
    char* sc = llm__json_escape(r->system);
    llm__buf_append(&b, ",\"preamble\":", 12);
    llm__buf_append(&b, sc, strlen(sc));
    free(sc);
  }
  if (r->messages_count > 0) {
    llm_message_t* last = &r->messages[r->messages_count - 1];
    if (last->content) {
      char* mc = llm__json_escape(last->content);
      llm__buf_append(&b, ",\"message\":", 11);
      llm__buf_append(&b, mc, strlen(mc));
      free(mc);
    }
  }
  if (r->messages_count > 1) {
    llm__buf_append(&b, ",\"chat_history\":[", 17);
    for (int i = 0; i < r->messages_count - 1; i++) {
      llm_message_t* m = &r->messages[i];
      if (i)
        llm__buf_append(&b, ",", 1);
      const char* rs = (m->role == LLM_ROLE_ASSISTANT) ? "CHATBOT" : "USER";
      char* ct = llm__json_escape(m->content ? m->content : "");
      llm__buf_append(&b, "{\"role\":\"", 9);
      llm__buf_append(&b, rs, strlen(rs));
      llm__buf_append(&b, "\",\"message\":", 12);
      llm__buf_append(&b, ct, strlen(ct));
      llm__buf_append(&b, "}", 1);
      free(ct);
    }
    llm__buf_append(&b, "]", 1);
  }
  llm__buf_append(&b, "}", 1);
  char* result = llm__strdup(b.buffer);
  llm__buf_free(&b);
  return result;
}

static llm_finish_reason_t llm__parse_finish_reason(const char* s) {
  if (!s)
    return LLM_FINISH_UNKNOWN;
  if (strcmp(s, "stop") == 0 || strcmp(s, "end_turn") == 0)
    return LLM_FINISH_STOP;
  if (strcmp(s, "length") == 0 || strcmp(s, "max_tokens") == 0)
    return LLM_FINISH_LENGTH;
  if (strcmp(s, "tool_calls") == 0 || strcmp(s, "tool_use") == 0)
    return LLM_FINISH_TOOL_CALL;
  if (strcmp(s, "content_filter") == 0)
    return LLM_FINISH_CONTENT_FILTER;
  return LLM_FINISH_UNKNOWN;
}

static llm_error_t llm__http_status_to_error(long s) {
  if (s == 200 || s == 201)
    return LLM_OK;
  if (s == 401 || s == 403)
    return LLM_ERR_AUTH;
  if (s == 404)
    return LLM_ERR_NOT_FOUND;
  if (s == 429)
    return LLM_ERR_RATE_LIMIT;
  if (s == 413)
    return LLM_ERR_CONTEXT_LENGTH;
  if (s >= 500)
    return LLM_ERR_SERVER;
  return LLM_ERR_HTTP;
}

static void llm__parse_openai_response(const char* body, llm_response_t* resp) {
  resp->id = llm__json_get_string(body, "id");
  resp->model = llm__json_get_string(body, "model");
  char* eo = llm__json_get_nested(body, "error");
  if (eo) {
    resp->error_message = llm__json_get_string(eo, "message");
    free(eo);
    return;
  }
  char* choices = llm__json_get_nested(body, "choices");
  if (!choices)
    return;
  char* c0 = llm__json_array_get(choices, 0);
  free(choices);
  if (!c0)
    return;
  char* fin = llm__json_get_string(c0, "finish_reason");
  if (fin) {
    resp->finish_reason = llm__parse_finish_reason(fin);
    free(fin);
  }
  char* msg = llm__json_get_nested(c0, "message");
  free(c0);
  if (!msg)
    return;
  resp->content = llm__json_get_string(msg, "content");
  char* tcs = llm__json_get_nested(msg, "tool_calls");
  if (tcs) {
    int n = llm__json_array_len(tcs);
    if (n > 0) {
      resp->tool_calls = (llm_tool_call_t*)calloc(n, sizeof(llm_tool_call_t));
      if (resp->tool_calls) {
        resp->tool_calls_count = n;
        for (int i = 0; i < n; i++) {
          char* tc = llm__json_array_get(tcs, i);
          if (!tc)
            continue;
          resp->tool_calls[i].id = llm__json_get_string(tc, "id");
          char* func = llm__json_get_nested(tc, "function");
          if (func) {
            resp->tool_calls[i].name = llm__json_get_string(func, "name");
            resp->tool_calls[i].arguments_json =
                llm__json_get_string(func, "arguments");
            free(func);
          }
          free(tc);
        }
      }
    }
    free(tcs);
  }
  free(msg);
  char* uobj = llm__json_get_nested(body, "usage");
  if (uobj) {
    resp->usage.prompt_tokens = (int)llm__json_get_int(uobj, "prompt_tokens");
    resp->usage.completion_tokens =
        (int)llm__json_get_int(uobj, "completion_tokens");
    resp->usage.total_tokens = (int)llm__json_get_int(uobj, "total_tokens");
    if (!resp->usage.total_tokens)
      resp->usage.total_tokens =
          resp->usage.prompt_tokens + resp->usage.completion_tokens;
    resp->usage.provided =
        (resp->usage.prompt_tokens > 0 || resp->usage.completion_tokens > 0);
    free(uobj);
  }
}

static void llm__parse_anthropic_response(const char* body,
                                          llm_response_t* resp) {
  resp->id = llm__json_get_string(body, "id");
  resp->model = llm__json_get_string(body, "model");
  char* eo = llm__json_get_nested(body, "error");
  if (eo) {
    resp->error_message = llm__json_get_string(eo, "message");
    free(eo);
    return;
  }
  char* sr = llm__json_get_string(body, "stop_reason");
  if (sr) {
    resp->finish_reason = llm__parse_finish_reason(sr);
    free(sr);
  }
  char* content = llm__json_get_nested(body, "content");
  if (!content)
    return;
  int n = llm__json_array_len(content);
  llm__buf_t tb = llm__buf_new();
  int tc = 0;
  llm_tool_call_t* tcs = NULL;
  for (int i = 0; i < n; i++) {
    char* item = llm__json_array_get(content, i);
    if (!item)
      continue;
    char* type = llm__json_get_string(item, "type");
    if (!type) {
      free(item);
      continue;
    }
    if (strcmp(type, "text") == 0) {
      char* text = llm__json_get_string(item, "text");
      if (text) {
        if (tb.size > 0)
          llm__buf_append(&tb, "\n", 1);
        llm__buf_append(&tb, text, strlen(text));
        free(text);
      }
    } else if (strcmp(type, "tool_use") == 0) {
      llm_tool_call_t* tmp2 =
          (llm_tool_call_t*)realloc(tcs, (tc + 1) * sizeof(llm_tool_call_t));
      if (tmp2) {
        tcs = tmp2;
        memset(&tcs[tc], 0, sizeof(llm_tool_call_t));
        tcs[tc].id = llm__json_get_string(item, "id");
        tcs[tc].name = llm__json_get_string(item, "name");
        char* inp = llm__json_get_nested(item, "input");
        if (inp)
          tcs[tc].arguments_json = inp;
        tc++;
      }
    }
    free(type);
    free(item);
  }
  free(content);
  if (tb.size > 0)
    resp->content = llm__strdup(tb.buffer);
  llm__buf_free(&tb);
  resp->tool_calls = tcs;
  resp->tool_calls_count = tc;
  char* uobj = llm__json_get_nested(body, "usage");
  if (uobj) {
    resp->usage.prompt_tokens = (int)llm__json_get_int(uobj, "input_tokens");
    resp->usage.completion_tokens =
        (int)llm__json_get_int(uobj, "output_tokens");
    resp->usage.total_tokens =
        resp->usage.prompt_tokens + resp->usage.completion_tokens;
    resp->usage.provided =
        (resp->usage.prompt_tokens > 0 || resp->usage.completion_tokens > 0);
    free(uobj);
  }
}

static void llm__parse_cohere_response(const char* body, llm_response_t* resp) {
  resp->id = llm__json_get_string(body, "generation_id");
  resp->content = llm__json_get_string(body, "text");
  char* fin = llm__json_get_string(body, "finish_reason");
  if (fin) {
    if (strcmp(fin, "COMPLETE") == 0)
      resp->finish_reason = LLM_FINISH_STOP;
    else if (strcmp(fin, "MAX_TOKENS") == 0)
      resp->finish_reason = LLM_FINISH_LENGTH;
    else
      resp->finish_reason = LLM_FINISH_UNKNOWN;
    free(fin);
  }
  char* meta = llm__json_get_nested(body, "meta");
  if (meta) {
    char* tokens = llm__json_get_nested(meta, "tokens");
    if (tokens) {
      resp->usage.prompt_tokens =
          (int)llm__json_get_int(tokens, "input_tokens");
      resp->usage.completion_tokens =
          (int)llm__json_get_int(tokens, "output_tokens");
      resp->usage.total_tokens =
          resp->usage.prompt_tokens + resp->usage.completion_tokens;
      resp->usage.provided =
          (resp->usage.prompt_tokens > 0 || resp->usage.completion_tokens > 0);
      free(tokens);
    }
    free(meta);
  }
}

typedef struct {
  llm_stream_callback_t cb;
  void* user_data;
  llm_provider_t provider;
  llm__buf_t leftover;
  double* first_token_time_out;
  double request_start_ms;
  int* chunk_count_out;
  llm__buf_t* error_buf_unused;
  bool* is_error_unused;
} llm__stream_ctx_t;

static void llm__stream_record_first(llm__stream_ctx_t* ctx) {
  if (ctx->first_token_time_out && *ctx->first_token_time_out < 0.0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *ctx->first_token_time_out =
        (ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6) - ctx->request_start_ms;
  }
  if (ctx->chunk_count_out)
    (*ctx->chunk_count_out)++;
}

static size_t llm__stream_write(void* data, size_t size, size_t nmemb,
                                void* userp) {
  size_t real = size * nmemb;
  llm__stream_ctx_t* ctx = (llm__stream_ctx_t*)userp;
  llm__buf_append(&ctx->leftover, (char*)data, real);
  char* buf = ctx->leftover.buffer;
  char* ls = buf;
  while (true) {
    char* nl = strchr(ls, '\n');
    if (!nl)
      break;
    *nl = '\0';
    char* line = ls;
    ls = nl + 1;
    while (*line == '\r')
      line++;
    if (*line == '\0')
      continue;

    if (ctx->provider == LLM_PROVIDER_ANTHROPIC) {
      if (strncmp(line, "data: ", 6) == 0) {
        const char* json = line + 6;
        char* type = llm__json_get_string(json, "type");
        if (type) {
          if (strcmp(type, "content_block_delta") == 0) {
            char* dobj = llm__json_get_nested(json, "delta");
            if (dobj) {
              char* dt = llm__json_get_string(dobj, "text");
              if (dt && strlen(dt) > 0) {
                llm__stream_record_first(ctx);
                if (ctx->cb)
                  ctx->cb(dt, false, ctx->user_data);
              }
              free(dt);
              free(dobj);
            }
          } else if (strcmp(type, "message_stop") == 0) {
            if (ctx->cb)
              ctx->cb(NULL, true, ctx->user_data);
          }
          free(type);
        }
      }
    } else if (ctx->provider == LLM_PROVIDER_COHERE) {
      if (strncmp(line, "data: ", 6) == 0) {
        const char* json = line + 6;
        char* et = llm__json_get_string(json, "event_type");
        if (et) {
          if (strcmp(et, "text-generation") == 0) {
            char* text = llm__json_get_string(json, "text");
            if (text) {
              llm__stream_record_first(ctx);
              if (ctx->cb)
                ctx->cb(text, false, ctx->user_data);
              free(text);
            }
          } else if (strcmp(et, "stream-end") == 0) {
            if (ctx->cb)
              ctx->cb(NULL, true, ctx->user_data);
          }
          free(et);
        }
      }
    } else {
      if (strncmp(line, "data: ", 6) == 0) {
        const char* json = line + 6;
        if (strcmp(json, "[DONE]") == 0) {
          if (ctx->cb)
            ctx->cb(NULL, true, ctx->user_data);
          continue;
        }
        char* choices = llm__json_get_nested(json, "choices");
        if (choices) {
          char* c0 = llm__json_array_get(choices, 0);
          free(choices);
          if (c0) {
            char* delta = llm__json_get_nested(c0, "delta");
            free(c0);
            if (delta) {
              char* content = llm__json_get_string(delta, "content");
              free(delta);
              if (content && strlen(content) > 0) {
                llm__stream_record_first(ctx);
                if (ctx->cb)
                  ctx->cb(content, false, ctx->user_data);
              }
              free(content);
            }
          }
        }
      }
    }
  }
  size_t remaining = strlen(ls);
  if (remaining > 0 && ls != buf) {
    memmove(buf, ls, remaining + 1);
    ctx->leftover.size = remaining;
  } else if (ls == buf + ctx->leftover.size) {
    ctx->leftover.size = 0;
    if (ctx->leftover.buffer)
      ctx->leftover.buffer[0] = '\0';
  }
  return real;
}

static bool llm__should_retry(long s, int attempt, int max_retries,
                              bool retry_on_rate_limit) {
  if (attempt >= max_retries)
    return false;
  if (s == 429 && retry_on_rate_limit)
    return true;
  if (s >= 500)
    return true;
  return false;
}

static int llm__retry_delay_ms(int attempt, int base, int maxd, double mult) {
  double d = base * pow(mult, (double)attempt);
  if (d > maxd)
    d = maxd;
  return (int)d;
}

static void llm__ms_sleep(int ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000L;
  nanosleep(&ts, NULL);
}

static double llm__now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static llm_error_t llm__do_request(llm_client_t* c, llm_request_t* r,
                                   llm__buf_t* rbuf, long* status_out,
                                   double start_ms, double* ftt_out,
                                   int* chunks_out) {
  CURL* curl = curl_easy_init();
  if (!curl)
    return LLM_ERR_CURL;
  llm_provider_t prov = c->provider;
  bool use_oai = llm__provider_uses_openai_compat(prov);
  bool is_cohere = (prov == LLM_PROVIDER_COHERE);
  const char* base_url =
      c->base_url ? c->base_url : llm__provider_default_url(prov);
  if (!base_url) {
    curl_easy_cleanup(curl);
    return LLM_ERR_INVALID_PARAM;
  }
  char url[1024];
  if (is_cohere)
    snprintf(url, sizeof(url), "%s/chat", base_url);
  else if (use_oai) {
    if (strstr(base_url, "/chat/completions"))
      snprintf(url, sizeof(url), "%s", base_url);
    else
      snprintf(url, sizeof(url), "%s/chat/completions", base_url);
  } else {
    if (strstr(base_url, "/messages"))
      snprintf(url, sizeof(url), "%s", base_url);
    else
      snprintf(url, sizeof(url), "%s/messages", base_url);
  }
  char* body = is_cohere ? llm__build_cohere_request(c, r)
                         : (!use_oai ? llm__build_anthropic_request(c, r)
                                     : llm__build_openai_request(c, r));
  if (!body) {
    curl_easy_cleanup(curl);
    return LLM_ERR_ALLOC;
  }
  LLM__LOG_DEBUG(c, "POST %s", url);
  LLM__LOG_TRACE(c, "Request body: %s", body);
  struct curl_slist* headers = NULL;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");
  char ah[512];
  if (c->api_key) {
    if (prov == LLM_PROVIDER_ANTHROPIC) {
      snprintf(ah, sizeof(ah), "x-api-key: %s", c->api_key);
      headers = curl_slist_append(headers, ah);
      headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
    } else {
      snprintf(ah, sizeof(ah), "Authorization: Bearer %s", c->api_key);
      headers = curl_slist_append(headers, ah);
    }
  }
  if (c->org_id) {
    char oh[256];
    snprintf(oh, sizeof(oh), "OpenAI-Organization: %s", c->org_id);
    headers = curl_slist_append(headers, oh);
  }
  if (c->project_id) {
    char ph[256];
    snprintf(ph, sizeof(ph), "OpenAI-Project: %s", c->project_id);
    headers = curl_slist_append(headers, ph);
  }
  for (int i = 0; i < c->extra_headers_count; i++) {
    char hdr[1024];
    snprintf(hdr, sizeof(hdr), "%s: %s", c->extra_headers[i].key,
             c->extra_headers[i].value);
    headers = curl_slist_append(headers, hdr);
  }
  llm__stream_ctx_t sctx;
  memset(&sctx, 0, sizeof(sctx));
  sctx.cb = r->stream_callback;
  sctx.user_data = r->stream_user_data ? r->stream_user_data : r->user_data;
  sctx.provider = prov;
  sctx.leftover = llm__buf_new();
  sctx.first_token_time_out = ftt_out;
  sctx.request_start_ms = start_ms;
  sctx.chunk_count_out = chunks_out;
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, c->timeout_seconds);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, c->verify_ssl ? 1L : 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, c->verify_ssl ? 2L : 0L);
  if (c->proxy)
    curl_easy_setopt(curl, CURLOPT_PROXY, c->proxy);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, llm__curl_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, rbuf);
  CURLcode res = curl_easy_perform(curl);
  long http_status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  free(body);
  if (status_out)
    *status_out = http_status;
  LLM__LOG_DEBUG(c, "HTTP status: %ld", http_status);
  if (rbuf->buffer)
    LLM__LOG_TRACE(c, "Response body: %s", rbuf->buffer);
  if (res == CURLE_OPERATION_TIMEDOUT)
    return LLM_ERR_TIMEOUT;
  if (res != CURLE_OK)
    return LLM_ERR_CURL;
  if (r->stream && (http_status == 200 || http_status == 201) && rbuf->buffer &&
      rbuf->size > 0) {
    const char* p = rbuf->buffer;
    while (*p) {
      const char* nl = strchr(p, '\n');
      size_t llen = nl ? (size_t)(nl - p) : strlen(p);
      char line[4096];
      if (llen >= sizeof(line))
        llen = sizeof(line) - 1;
      memcpy(line, p, llen);
      line[llen] = '\0';
      char* l = line;
      while (*l == '\r')
        l++;
      if (*l != '\0' && strncmp(l, "data: ", 6) == 0) {
        const char* json = l + 6;
        if (strcmp(json, "[DONE]") == 0) {
          if (sctx.cb)
            sctx.cb(NULL, true, sctx.user_data);
        } else if (prov == LLM_PROVIDER_ANTHROPIC) {
          char* type = llm__json_get_string(json, "type");
          if (type) {
            if (strcmp(type, "content_block_delta") == 0) {
              char* dobj = llm__json_get_nested(json, "delta");
              if (dobj) {
                char* dt = llm__json_get_string(dobj, "text");
                if (dt && strlen(dt) > 0) {
                  llm__stream_record_first(&sctx);
                  if (sctx.cb)
                    sctx.cb(dt, false, sctx.user_data);
                }
                free(dt);
                free(dobj);
              }
            } else if (strcmp(type, "message_stop") == 0) {
              if (sctx.cb)
                sctx.cb(NULL, true, sctx.user_data);
            }
            free(type);
          }
        } else if (prov == LLM_PROVIDER_COHERE) {
          char* et = llm__json_get_string(json, "event_type");
          if (et) {
            if (strcmp(et, "text-generation") == 0) {
              char* text = llm__json_get_string(json, "text");
              if (text) {
                llm__stream_record_first(&sctx);
                if (sctx.cb)
                  sctx.cb(text, false, sctx.user_data);
                free(text);
              }
            } else if (strcmp(et, "stream-end") == 0) {
              if (sctx.cb)
                sctx.cb(NULL, true, sctx.user_data);
            }
            free(et);
          }
        } else {
          char* choices = llm__json_get_nested(json, "choices");
          if (choices) {
            char* c0 = llm__json_array_get(choices, 0);
            free(choices);
            if (c0) {
              char* delta = llm__json_get_nested(c0, "delta");
              free(c0);
              if (delta) {
                char* content = llm__json_get_string(delta, "content");
                free(delta);
                if (content && strlen(content) > 0) {
                  llm__stream_record_first(&sctx);
                  if (sctx.cb)
                    sctx.cb(content, false, sctx.user_data);
                }
                free(content);
              }
            }
          }
        }
      }
      if (!nl)
        break;
      p = nl + 1;
    }
  }
  llm__buf_free(&sctx.leftover);
  return LLM_OK;
}

static llm_response_t* llm_complete(llm_client_t* c, llm_request_t* r) {
  llm_response_t* resp = (llm_response_t*)calloc(1, sizeof(llm_response_t));
  if (!resp)
    return NULL;
  resp->user_data = r->user_data;
  int attempt = 0;
  double start_ms = llm__now_ms();
  double ftt = -1.0;
  int chunks = 0;
  while (true) {
    llm__buf_t rbuf = llm__buf_new();
    long http_status = 0;
    LLM__LOG_INFO(c, "Request attempt %d/%d", attempt + 1, c->max_retries + 1);
    llm_error_t err =
        llm__do_request(c, r, &rbuf, &http_status, start_ms, &ftt, &chunks);
    resp->http_status = (int)http_status;
    if (err == LLM_ERR_TIMEOUT) {
      llm__buf_free(&rbuf);
      if (attempt < c->max_retries) {
        int d = llm__retry_delay_ms(attempt, c->retry_delay_ms,
                                    c->retry_delay_max_ms,
                                    c->retry_backoff_multiplier);
        LLM__LOG_INFO(c, "Timeout, retrying in %dms", d);
        llm__ms_sleep(d);
        attempt++;
        continue;
      }
      resp->error = LLM_ERR_TIMEOUT;
      resp->error_message = llm__strdup("Request timed out");
      break;
    }
    if (err == LLM_ERR_CURL) {
      llm__buf_free(&rbuf);
      resp->error = LLM_ERR_CURL;
      resp->error_message = llm__strdup("CURL error");
      break;
    }
    if (llm__should_retry(http_status, attempt, c->max_retries,
                          c->retry_on_rate_limit)) {
      int d =
          llm__retry_delay_ms(attempt, c->retry_delay_ms, c->retry_delay_max_ms,
                              c->retry_backoff_multiplier);
      LLM__LOG_INFO(c, "HTTP %ld, retrying in %dms", http_status, d);
      llm__buf_free(&rbuf);
      llm__ms_sleep(d);
      attempt++;
      continue;
    }
    if (http_status != 200 && http_status != 201 &&
        !(r->stream && http_status == 0)) {
      resp->error = llm__http_status_to_error(http_status);
      if (rbuf.buffer) {
        resp->error_message = llm__json_get_string(rbuf.buffer, "message");
        if (!resp->error_message) {
          char* eo = llm__json_get_nested(rbuf.buffer, "error");
          if (eo) {
            resp->error_message = llm__json_get_string(eo, "message");
            free(eo);
          }
        }
        if (!resp->error_message)
          resp->error_message = llm__strdup(rbuf.buffer);
      }
      llm__buf_free(&rbuf);
      break;
    }
    if (!r->stream) {
      if (c->provider == LLM_PROVIDER_COHERE)
        llm__parse_cohere_response(rbuf.buffer, resp);
      else if (c->provider == LLM_PROVIDER_ANTHROPIC)
        llm__parse_anthropic_response(rbuf.buffer, resp);
      else
        llm__parse_openai_response(rbuf.buffer, resp);
    }
    if (!resp->model && c->model)
      resp->model = llm__strdup(c->model);
    llm__buf_free(&rbuf);
    break;
  }
  resp->stats.latency_ms = llm__now_ms() - start_ms;
  resp->stats.retries = attempt;
  resp->stats.stream_chunks = chunks;
  if (ftt >= 0.0) {
    resp->stats.time_to_first_token_ms = ftt;
    resp->stats.has_time_to_first_token = true;
  }
  LLM__LOG_INFO(c, "Completed in %.1fms, retries=%d", resp->stats.latency_ms,
                attempt);
  if (resp->usage.provided)
    LLM__LOG_INFO(c, "Tokens: prompt=%d completion=%d total=%d",
                  resp->usage.prompt_tokens, resp->usage.completion_tokens,
                  resp->usage.total_tokens);
  return resp;
}

typedef struct {
  llm_client_t* client;
  llm_request_t* request;
  llm_async_callback_t callback;
  void* user_data;
} llm__async_arg_t;

static void* llm__async_thread(void* arg) {
  llm__async_arg_t* a = (llm__async_arg_t*)arg;
  llm_response_t* resp = llm_complete(a->client, a->request);
  if (resp)
    resp->user_data = a->user_data;
  if (a->callback)
    a->callback(resp, a->user_data);
  free(a);
  return NULL;
}

static llm_error_t llm_complete_async(llm_client_t* c, llm_request_t* r,
                                      llm_async_callback_t cb,
                                      void* user_data) {
  llm__async_arg_t* arg = (llm__async_arg_t*)malloc(sizeof(llm__async_arg_t));
  if (!arg)
    return LLM_ERR_ALLOC;
  arg->client = c;
  arg->request = r;
  arg->callback = cb;
  arg->user_data = user_data;
  pthread_t tid;
  if (pthread_create(&tid, NULL, llm__async_thread, arg) != 0) {
    free(arg);
    return LLM_ERR_THREAD;
  }
  pthread_detach(tid);
  return LLM_OK;
}

typedef struct {
  llm_client_t* client;
  llm_request_t* request;
  llm_response_t** result;
  pthread_mutex_t* mutex;
  pthread_cond_t* cond;
  int* remaining;
  llm_async_callback_t callback;
  void* user_data;
} llm__batch_arg_t;

static void* llm__batch_thread(void* arg) {
  llm__batch_arg_t* a = (llm__batch_arg_t*)arg;
  llm_response_t* resp = llm_complete(a->client, a->request);
  if (resp)
    resp->user_data = a->user_data;
  if (a->callback)
    a->callback(resp, a->user_data);
  if (a->result)
    *a->result = resp;
  pthread_mutex_lock(a->mutex);
  (*a->remaining)--;
  pthread_cond_signal(a->cond);
  pthread_mutex_unlock(a->mutex);
  free(a);
  return NULL;
}

static llm_response_t** llm_complete_batch(llm_client_t* c,
                                           llm_request_t* requests, int count,
                                           llm_async_callback_t per_cb,
                                           void* user_data,
                                           int max_concurrent) {
  if (count <= 0 || !requests)
    return NULL;
  llm_response_t** results =
      (llm_response_t**)calloc(count, sizeof(llm_response_t*));
  if (!results)
    return NULL;
  if (max_concurrent <= 0)
    max_concurrent = count;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);
  int remaining = count, started = 0, in_flight = 0;
  pthread_mutex_lock(&mutex);
  while (started < count || remaining > 0) {
    while (started < count && in_flight < max_concurrent) {
      llm__batch_arg_t* barg =
          (llm__batch_arg_t*)malloc(sizeof(llm__batch_arg_t));
      if (!barg) {
        started++;
        remaining--;
        continue;
      }
      barg->client = c;
      barg->request = &requests[started];
      barg->result = &results[started];
      barg->mutex = &mutex;
      barg->cond = &cond;
      barg->remaining = &remaining;
      barg->callback = per_cb;
      barg->user_data = user_data;
      pthread_t tid;
      if (pthread_create(&tid, NULL, llm__batch_thread, barg) == 0) {
        pthread_detach(tid);
        in_flight++;
        LLM__LOG_DEBUG(c, "Batch: started %d/%d", started + 1, count);
      } else {
        free(barg);
        remaining--;
      }
      started++;
    }
    if (remaining > 0) {
      int prev = remaining;
      pthread_cond_wait(&cond, &mutex);
      in_flight -= (prev - remaining);
    } else
      break;
  }
  pthread_mutex_unlock(&mutex);
  pthread_mutex_destroy(&mutex);
  pthread_cond_destroy(&cond);
  return results;
}

static void llm_batch_free(llm_response_t** responses, int count) {
  if (!responses)
    return;
  for (int i = 0; i < count; i++) {
    if (responses[i]) {
      llm_response_free(responses[i]);
      free(responses[i]);
    }
  }
  free(responses);
}

static const char* llm_error_string(llm_error_t err) {
  switch (err) {
  case LLM_OK:
    return "ok";
  case LLM_ERR_INVALID_PARAM:
    return "invalid_param";
  case LLM_ERR_ALLOC:
    return "alloc";
  case LLM_ERR_CURL:
    return "curl";
  case LLM_ERR_HTTP:
    return "http";
  case LLM_ERR_PARSE:
    return "parse";
  case LLM_ERR_TIMEOUT:
    return "timeout";
  case LLM_ERR_RATE_LIMIT:
    return "rate_limit";
  case LLM_ERR_AUTH:
    return "auth";
  case LLM_ERR_NOT_FOUND:
    return "not_found";
  case LLM_ERR_SERVER:
    return "server";
  case LLM_ERR_UNSUPPORTED:
    return "unsupported";
  case LLM_ERR_CANCELLED:
    return "cancelled";
  case LLM_ERR_THREAD:
    return "thread";
  case LLM_ERR_CONTEXT_LENGTH:
    return "context_length";
  default:
    return "unknown";
  }
}

static const char* llm_provider_name(llm_provider_t p) {
  switch (p) {
  case LLM_PROVIDER_OPENAI:
    return "openai";
  case LLM_PROVIDER_ANTHROPIC:
    return "anthropic";
  case LLM_PROVIDER_GROQ:
    return "groq";
  case LLM_PROVIDER_OLLAMA:
    return "ollama";
  case LLM_PROVIDER_TOGETHER:
    return "together";
  case LLM_PROVIDER_MISTRAL:
    return "mistral";
  case LLM_PROVIDER_COHERE:
    return "cohere";
  case LLM_PROVIDER_GEMINI:
    return "gemini";
  case LLM_PROVIDER_DEEPSEEK:
    return "deepseek";
  case LLM_PROVIDER_OPENROUTER:
    return "openrouter";
  case LLM_PROVIDER_PERPLEXITY:
    return "perplexity";
  case LLM_PROVIDER_FIREWORKS:
    return "fireworks";
  case LLM_PROVIDER_VLLM:
    return "vllm";
  case LLM_PROVIDER_CUSTOM:
    return "custom";
  default:
    return "unknown";
  }
}

static llm_provider_t llm_provider_from_string(const char* s) {
  if (!s)
    return LLM_PROVIDER_OPENAI;
  if (strcmp(s, "openai") == 0)
    return LLM_PROVIDER_OPENAI;
  if (strcmp(s, "anthropic") == 0)
    return LLM_PROVIDER_ANTHROPIC;
  if (strcmp(s, "groq") == 0)
    return LLM_PROVIDER_GROQ;
  if (strcmp(s, "ollama") == 0)
    return LLM_PROVIDER_OLLAMA;
  if (strcmp(s, "together") == 0)
    return LLM_PROVIDER_TOGETHER;
  if (strcmp(s, "mistral") == 0)
    return LLM_PROVIDER_MISTRAL;
  if (strcmp(s, "cohere") == 0)
    return LLM_PROVIDER_COHERE;
  if (strcmp(s, "gemini") == 0)
    return LLM_PROVIDER_GEMINI;
  if (strcmp(s, "deepseek") == 0)
    return LLM_PROVIDER_DEEPSEEK;
  if (strcmp(s, "openrouter") == 0)
    return LLM_PROVIDER_OPENROUTER;
  if (strcmp(s, "perplexity") == 0)
    return LLM_PROVIDER_PERPLEXITY;
  if (strcmp(s, "fireworks") == 0)
    return LLM_PROVIDER_FIREWORKS;
  if (strcmp(s, "vllm") == 0)
    return LLM_PROVIDER_VLLM;
  return LLM_PROVIDER_CUSTOM;
}

static llm_message_t llm_msg_user(const char* content) {
  llm_message_t m = {0};
  m.role = LLM_ROLE_USER;
  m.content = llm__strdup(content);
  return m;
}

static llm_message_t llm_msg_assistant(const char* content) {
  llm_message_t m = {0};
  m.role = LLM_ROLE_ASSISTANT;
  m.content = llm__strdup(content);
  return m;
}

static llm_message_t llm_msg_system(const char* content) {
  llm_message_t m = {0};
  m.role = LLM_ROLE_SYSTEM;
  m.content = llm__strdup(content);
  return m;
}

static llm_message_t llm_msg_tool_result(const char* tool_call_id,
                                         const char* content) {
  llm_message_t m = {0};
  m.role = LLM_ROLE_TOOL;
  m.tool_call_id = llm__strdup(tool_call_id);
  m.content = llm__strdup(content);
  return m;
}

static llm_tool_def_t llm_tool_function(const char* name, const char* desc,
                                        const char* params_json) {
  llm_tool_def_t t = {0};
  t.type = LLM_TOOL_TYPE_FUNCTION;
  t.function.name = llm__strdup(name);
  t.function.description = llm__strdup(desc);
  t.function.parameters_json = params_json ? llm__strdup(params_json) : NULL;
  return t;
}

static void llm_tool_def_free(llm_tool_def_t* t) {
  free(t->function.name);
  free(t->function.description);
  free(t->function.parameters_json);
  memset(t, 0, sizeof(*t));
}

static llm_error_t llm_client_add_header(llm_client_t* c, const char* key,
                                         const char* value) {
  llm_header_t* nh = (llm_header_t*)realloc(
      c->extra_headers, (c->extra_headers_count + 1) * sizeof(llm_header_t));
  if (!nh)
    return LLM_ERR_ALLOC;
  c->extra_headers = nh;
  c->extra_headers[c->extra_headers_count].key = llm__strdup(key);
  c->extra_headers[c->extra_headers_count].value = llm__strdup(value);
  c->extra_headers_count++;
  return LLM_OK;
}

static void llm_print_stats(llm_response_t* resp, FILE* out) {
  if (!resp || !out)
    return;
  fprintf(out, "=== stats ===\n");
  fprintf(out, "latency_ms: %.2f\n", resp->stats.latency_ms);
  fprintf(out, "retries: %d\n", resp->stats.retries);
  fprintf(out, "prompt_tokens: %d\n", resp->usage.prompt_tokens);
  fprintf(out, "completion_tokens: %d\n", resp->usage.completion_tokens);
  fprintf(out, "total_tokens: %d\n", resp->usage.total_tokens);
  fprintf(out, "token_usage_provided: %s\n",
          resp->usage.provided ? "yes" : "no");
  if (resp->stats.has_time_to_first_token)
    fprintf(out, "time_to_first_token_ms: %.2f\n",
            resp->stats.time_to_first_token_ms);
  else
    fprintf(out, "time_to_first_token_ms: n/a\n");
  fprintf(out, "stream_chunks: %d\n", resp->stats.stream_chunks);
  fprintf(out, "error: %s\n",
          resp->error != LLM_OK ? llm_error_string(resp->error) : "none");
}

static void llm_init(void) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
}

static void llm_cleanup(void) {
  curl_global_cleanup();
}

#ifdef __cplusplus
}
#endif
#endif
