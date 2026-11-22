#pragma once

// Repetition guard for agent loops: detects a run of near-identical trivial tool calls at the end of a chat (the model
// keeps calling the same tool with almost the same arguments and gets short results back, e.g. "echo end1" .. "echo
// end55") so the server can add one short note to that request. Stateless: it only reads the messages of the request.

#include "chat.h"

#include <string>
#include <vector>

struct common_repeat_guard_result {
    bool        fire = false;
    int         rule = 0;     // 1 = near-identical arguments, 2 = a long run of short-result calls to one tool
    int         count = 0;    // calls to the tool in the trailing run
    std::string tool;         // the repeated tool's name
};

// Only fires when the last message is a tool result. The trailing run ends at a user message or at an assistant turn
// without tool calls.
//   rule 1: the last 6 assistant turns each made exactly one call to the same tool, each call's arguments are within a
//           small edit of the previous one (digits ignored, similarity >= 0.8), and every result is <= 200 characters
//   rule 2: the trailing run of calls to one tool with results <= 200 characters (a longer result or another tool ends
//           the run) has >= 8 calls over >= 4 assistant turns (parallel calls of one turn are one step), and that tool
//           does not change files (names with edit / write / patch / replace / create / insert / apply)
common_repeat_guard_result common_repeat_guard_check(const std::vector<common_chat_msg> & msgs, bool use_rule2 = true);

// The note the server appends as the newest message (never into the system prompt).
std::string common_repeat_guard_note(const common_repeat_guard_result & r);
