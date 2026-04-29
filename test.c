#include "llm.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

static int passed = 0;
static int failed = 0;

#define TEST(name) void test_##name(void)
#define RUN(name)                                                              \
  do {                                                                         \
    printf("  " #name " ... ");                                                \
    test_##name();                                                             \
  } while (0)
#define OK()                                                                   \
  do {                                                                         \
    printf("ok\n");                                                            \
    passed++;                                                                  \
  } while (0)
#define FAIL(msg)                                                              \
  do {                                                                         \
    printf("FAIL: %s\n", msg);                                                 \
    failed++;                                                                  \
  } while (0)
#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      FAIL(#cond);                                                             \
      return;                                                                  \
    }                                                                          \
  } while (0)

TEST(client_init_defaults) {
  llm_client_t c;
  llm_client_init(&c);
  ASSERT(c.timeout_seconds == LLM_DEFAULT_TIMEOUT);
  ASSERT(c.max_retries == LLM_DEFAULT_MAX_RETRIES);
  ASSERT(c.retry_delay_ms == LLM_DEFAULT_RETRY_DELAY_MS);
  ASSERT(c.retry_on_rate_limit == true);
  ASSERT(c.verify_ssl == true);
  ASSERT(c.verbosity == 0);
  ASSERT(c.api_key == NULL);
  ASSERT(c.model == NULL);
  OK();
}

TEST(request_init_defaults) {
  llm_request_t r;
  llm_request_init(&r);
  ASSERT(r.max_tokens == LLM_DEFAULT_MAX_TOKENS);
  ASSERT(r.temperature == LLM_DEFAULT_TEMPERATURE);
  ASSERT(r.top_p == 1.0);
  ASSERT(r.messages_count == 0);
  ASSERT(r.stream == false);
  OK();
}

TEST(provider_from_string) {
  ASSERT(llm_provider_from_string("openai") == LLM_PROVIDER_OPENAI);
  ASSERT(llm_provider_from_string("anthropic") == LLM_PROVIDER_ANTHROPIC);
  ASSERT(llm_provider_from_string("groq") == LLM_PROVIDER_GROQ);
  ASSERT(llm_provider_from_string("ollama") == LLM_PROVIDER_OLLAMA);
  ASSERT(llm_provider_from_string("together") == LLM_PROVIDER_TOGETHER);
  ASSERT(llm_provider_from_string("mistral") == LLM_PROVIDER_MISTRAL);
  ASSERT(llm_provider_from_string("cohere") == LLM_PROVIDER_COHERE);
  ASSERT(llm_provider_from_string("gemini") == LLM_PROVIDER_GEMINI);
  ASSERT(llm_provider_from_string("deepseek") == LLM_PROVIDER_DEEPSEEK);
  ASSERT(llm_provider_from_string("openrouter") == LLM_PROVIDER_OPENROUTER);
  ASSERT(llm_provider_from_string("perplexity") == LLM_PROVIDER_PERPLEXITY);
  ASSERT(llm_provider_from_string("fireworks") == LLM_PROVIDER_FIREWORKS);
  ASSERT(llm_provider_from_string("vllm") == LLM_PROVIDER_VLLM);
  ASSERT(llm_provider_from_string("custom") == LLM_PROVIDER_CUSTOM);
  ASSERT(llm_provider_from_string("unknown_xyz") == LLM_PROVIDER_CUSTOM);
  OK();
}

TEST(provider_name) {
  ASSERT(strcmp(llm_provider_name(LLM_PROVIDER_OPENAI), "openai") == 0);
  ASSERT(strcmp(llm_provider_name(LLM_PROVIDER_ANTHROPIC), "anthropic") == 0);
  ASSERT(strcmp(llm_provider_name(LLM_PROVIDER_VLLM), "vllm") == 0);
  ASSERT(strcmp(llm_provider_name(LLM_PROVIDER_DEEPSEEK), "deepseek") == 0);
  ASSERT(strcmp(llm_provider_name(LLM_PROVIDER_GEMINI), "gemini") == 0);
  OK();
}

TEST(error_string) {
  ASSERT(strcmp(llm_error_string(LLM_OK), "ok") == 0);
  ASSERT(strcmp(llm_error_string(LLM_ERR_TIMEOUT), "timeout") == 0);
  ASSERT(strcmp(llm_error_string(LLM_ERR_RATE_LIMIT), "rate_limit") == 0);
  ASSERT(strcmp(llm_error_string(LLM_ERR_AUTH), "auth") == 0);
  ASSERT(strcmp(llm_error_string(LLM_ERR_SERVER), "server") == 0);
  ASSERT(strcmp(llm_error_string(LLM_ERR_CONTEXT_LENGTH), "context_length") ==
         0);
  OK();
}

TEST(message_helpers) {
  llm_message_t u = llm_msg_user("hello");
  ASSERT(u.role == LLM_ROLE_USER);
  ASSERT(strcmp(u.content, "hello") == 0);

  llm_message_t a = llm_msg_assistant("world");
  ASSERT(a.role == LLM_ROLE_ASSISTANT);
  ASSERT(strcmp(a.content, "world") == 0);

  llm_message_t s = llm_msg_system("be helpful");
  ASSERT(s.role == LLM_ROLE_SYSTEM);
  ASSERT(strcmp(s.content, "be helpful") == 0);

  llm_message_t t = llm_msg_tool_result("call_123", "{\"result\":1}");
  ASSERT(t.role == LLM_ROLE_TOOL);
  ASSERT(strcmp(t.tool_call_id, "call_123") == 0);
  ASSERT(strcmp(t.content, "{\"result\":1}") == 0);

  llm_message_free(&u);
  llm_message_free(&a);
  llm_message_free(&s);
  llm_message_free(&t);
  OK();
}

TEST(tool_function_helper) {
  llm_tool_def_t td = llm_tool_function("get_weather", "Returns weather",
                                        "{\"type\":\"object\"}");
  ASSERT(td.type == LLM_TOOL_TYPE_FUNCTION);
  ASSERT(strcmp(td.function.name, "get_weather") == 0);
  ASSERT(strcmp(td.function.description, "Returns weather") == 0);
  ASSERT(strcmp(td.function.parameters_json, "{\"type\":\"object\"}") == 0);
  llm_tool_def_free(&td);
  ASSERT(td.function.name == NULL);
  OK();
}

TEST(add_header) {
  llm_client_t c;
  llm_client_init(&c);
  llm_error_t err = llm_client_add_header(&c, "X-Foo", "bar");
  ASSERT(err == LLM_OK);
  ASSERT(c.extra_headers_count == 1);
  ASSERT(strcmp(c.extra_headers[0].key, "X-Foo") == 0);
  ASSERT(strcmp(c.extra_headers[0].value, "bar") == 0);
  err = llm_client_add_header(&c, "X-Baz", "qux");
  ASSERT(err == LLM_OK);
  ASSERT(c.extra_headers_count == 2);
  llm_client_free(&c);
  OK();
}

TEST(provider_uses_openai_compat) {
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_OPENAI) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_GROQ) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_OLLAMA) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_GEMINI) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_DEEPSEEK) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_VLLM) == true);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_ANTHROPIC) == false);
  ASSERT(llm__provider_uses_openai_compat(LLM_PROVIDER_COHERE) == false);
  OK();
}

TEST(provider_default_urls) {
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_OPENAI), "openai.com") !=
         NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_ANTHROPIC),
                "anthropic.com") != NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_GROQ), "groq.com") !=
         NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_OLLAMA), "11434") !=
         NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_GEMINI),
                "googleapis.com") != NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_DEEPSEEK),
                "deepseek.com") != NULL);
  ASSERT(strstr(llm__provider_default_url(LLM_PROVIDER_VLLM), "8000") != NULL);
  ASSERT(llm__provider_default_url(LLM_PROVIDER_CUSTOM) == NULL);
  OK();
}

TEST(response_free_null_safe) {
  llm_response_free(NULL);
  llm_response_t r = {0};
  llm_response_free(&r);
  OK();
}

TEST(json_escape) {
  char* s = llm__json_escape("hello \"world\"\nnewline\ttab");
  ASSERT(s != NULL);
  ASSERT(strstr(s, "\\\"") != NULL);
  ASSERT(strstr(s, "\\n") != NULL);
  ASSERT(strstr(s, "\\t") != NULL);
  free(s);

  char* n = llm__json_escape(NULL);
  ASSERT(strcmp(n, "null") == 0);
  free(n);
  OK();
}

TEST(json_get_string) {
  const char* json = "{\"name\":\"Alice\",\"city\":\"NYC\"}";
  char* name = llm__json_get_string(json, "name");
  ASSERT(name != NULL);
  ASSERT(strcmp(name, "Alice") == 0);
  free(name);

  char* city = llm__json_get_string(json, "city");
  ASSERT(city != NULL);
  ASSERT(strcmp(city, "NYC") == 0);
  free(city);

  char* miss = llm__json_get_string(json, "missing");
  ASSERT(miss == NULL);
  OK();
}

TEST(json_get_int) {
  const char* json = "{\"count\":42,\"total\":1000}";
  ASSERT(llm__json_get_int(json, "count") == 42);
  ASSERT(llm__json_get_int(json, "total") == 1000);
  ASSERT(llm__json_get_int(json, "missing") == 0);
  OK();
}

TEST(json_array_len) {
  ASSERT(llm__json_array_len("[]") == 0);
  ASSERT(llm__json_array_len("[1]") == 1);
  ASSERT(llm__json_array_len("[1,2,3]") == 3);
  ASSERT(llm__json_array_len("[{\"a\":1},{\"b\":2}]") == 2);
  ASSERT(llm__json_array_len(NULL) == 0);
  OK();
}

TEST(json_array_get) {
  char* item = llm__json_array_get("[\"a\",\"b\",\"c\"]", 1);
  ASSERT(item != NULL);
  ASSERT(strcmp(item, "\"b\"") == 0);
  free(item);

  char* none = llm__json_array_get("[\"a\"]", 5);
  ASSERT(none == NULL);
  OK();
}

TEST(buf_operations) {
  llm__buf_t b = llm__buf_new();
  ASSERT(b.buffer != NULL);
  ASSERT(b.size == 0);
  llm__buf_append(&b, "hello", 5);
  ASSERT(b.size == 5);
  ASSERT(strcmp(b.buffer, "hello") == 0);
  llm__buf_append(&b, " world", 6);
  ASSERT(strcmp(b.buffer, "hello world") == 0);
  llm__buf_free(&b);
  ASSERT(b.buffer == NULL);
  OK();
}

TEST(stats_struct_defaults) {
  llm_response_t r = {0};
  ASSERT(r.stats.latency_ms == 0.0);
  ASSERT(r.stats.retries == 0);
  ASSERT(r.stats.stream_chunks == 0);
  ASSERT(r.stats.has_time_to_first_token == false);
  ASSERT(r.usage.provided == false);
  OK();
}

int main(void) {
  llm_init();
  printf("Running tests...\n");
  RUN(client_init_defaults);
  RUN(request_init_defaults);
  RUN(provider_from_string);
  RUN(provider_name);
  RUN(error_string);
  RUN(message_helpers);
  RUN(tool_function_helper);
  RUN(add_header);
  RUN(provider_uses_openai_compat);
  RUN(provider_default_urls);
  RUN(response_free_null_safe);
  RUN(json_escape);
  RUN(json_get_string);
  RUN(json_get_int);
  RUN(json_array_len);
  RUN(json_array_get);
  RUN(buf_operations);
  RUN(stats_struct_defaults);
  printf("\n%d passed, %d failed\n", passed, failed);
  llm_cleanup();
  return failed > 0 ? 1 : 0;
}
