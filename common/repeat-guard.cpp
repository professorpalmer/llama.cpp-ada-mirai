#include "repeat-guard.h"

#include <algorithm>
#include <cctype>
#include <map>

static const int    RG_RULE1_TURNS   = 6;
static const int    RG_RULE2_CALLS   = 8;
static const size_t RG_SHORT_RESULT  = 200;
static const double RG_SIMILAR       = 0.8;
static const size_t RG_MAX_ARGS_CMP  = 2000;   // bound the edit-distance cost

static size_t rg_text_len(const common_chat_msg & m) {
    size_t n = m.content.size();
    for (const auto & p : m.content_parts) {
        n += p.text.size();
    }
    return n;
}

// arguments without digits and with runs of white space collapsed, so "echo end1" and "echo end55" compare equal
static std::string rg_norm(const std::string & s) {
    std::string out;
    out.reserve(std::min(s.size(), RG_MAX_ARGS_CMP));
    bool space = false;
    for (unsigned char c : s) {
        if (out.size() >= RG_MAX_ARGS_CMP) {
            break;
        }
        if (std::isdigit(c)) {
            continue;
        }
        if (std::isspace(c)) {
            space = true;
            continue;
        }
        if (space && !out.empty()) {
            out.push_back(' ');
        }
        space = false;
        out.push_back((char) c);
    }
    return out;
}

// 1 - Levenshtein distance / longer length (1.0 = equal)
static double rg_similarity(const std::string & a, const std::string & b) {
    if (a == b) {
        return 1.0;
    }
    const size_t n = a.size(), m = b.size();
    if (n == 0 || m == 0) {
        return 0.0;
    }
    std::vector<size_t> prev(m + 1), cur(m + 1);
    for (size_t j = 0; j <= m; ++j) {
        prev[j] = j;
    }
    for (size_t i = 1; i <= n; ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= m; ++j) {
            const size_t sub = prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            cur[j] = std::min({ prev[j] + 1, cur[j - 1] + 1, sub });
        }
        std::swap(prev, cur);
    }
    return 1.0 - (double) prev[m] / (double) std::max(n, m);
}

static bool rg_changes_files(const std::string & tool) {
    std::string t = tool;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    for (const char * w : { "edit", "write", "patch", "replace", "create", "insert", "apply" }) {
        if (t.find(w) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct rg_call {
    std::string name;
    std::string args;
    size_t      result_len = 0;
    bool        has_result = false;
};

common_repeat_guard_result common_repeat_guard_check(const std::vector<common_chat_msg> & msgs, bool use_rule2) {
    common_repeat_guard_result res;
    if (msgs.empty() || msgs.back().role != "tool") {
        return res;
    }
    // the trailing run: from the end back to a user message or an assistant turn without tool calls
    size_t start = msgs.size();
    while (start > 0) {
        const auto & m = msgs[start - 1];
        if (m.role == "user" || (m.role == "assistant" && m.tool_calls.empty())) {
            break;
        }
        --start;
    }
    // turns (one per assistant message with calls) and the calls in order, with their results
    std::vector<std::vector<rg_call>> turns;
    std::map<std::string, std::pair<size_t, size_t>> by_id;   // tool_call_id -> (turn, call)
    for (size_t i = start; i < msgs.size(); ++i) {
        const auto & m = msgs[i];
        if (m.role == "assistant") {
            turns.emplace_back();
            for (const auto & tc : m.tool_calls) {
                if (!tc.id.empty()) {
                    by_id[tc.id] = { turns.size() - 1, turns.back().size() };
                }
                turns.back().push_back({ tc.name, tc.arguments, 0, false });
            }
        } else if (m.role == "tool" && !turns.empty()) {
            rg_call * c = nullptr;
            auto it = m.tool_call_id.empty() ? by_id.end() : by_id.find(m.tool_call_id);
            if (it != by_id.end()) {
                c = &turns[it->second.first][it->second.second];
            } else {
                for (auto & tc : turns.back()) {   // no id: the first call of the latest turn without a result
                    if (!tc.has_result) {
                        c = &tc;
                        break;
                    }
                }
            }
            if (c) {
                c->result_len += rg_text_len(m);
                c->has_result = true;
            }
        }
    }
    if (turns.empty()) {
        return res;
    }
    std::vector<const rg_call *> calls;
    for (const auto & t : turns) {
        for (const auto & c : t) {
            calls.push_back(&c);
        }
    }
    if (calls.empty()) {
        return res;
    }
    const std::string tool = calls.back()->name;
    int run = 0;
    for (auto it = calls.rbegin(); it != calls.rend() && (*it)->name == tool; ++it) {
        ++run;
    }
    const auto short_result = [](const rg_call * c) { return c->has_result && c->result_len <= RG_SHORT_RESULT; };

    // rule 1: the last 6 turns, one call each, same tool, near-identical arguments, short results
    if ((int) turns.size() >= RG_RULE1_TURNS) {
        bool ok = true;
        std::string prev;
        for (size_t k = turns.size() - RG_RULE1_TURNS; k < turns.size() && ok; ++k) {
            const auto & t = turns[k];
            if (t.size() != 1 || t[0].name != tool || !short_result(&t[0])) {
                ok = false;
                break;
            }
            const std::string cur = rg_norm(t[0].args);
            if (k > turns.size() - RG_RULE1_TURNS && rg_similarity(prev, cur) < RG_SIMILAR) {
                ok = false;
            }
            prev = cur;
        }
        if (ok) {
            res.fire = true;
            res.rule = 1;
        }
    }
    // rule 2: >= 8 trailing calls to one tool that does not change files, all with short results
    if (!res.fire && use_rule2 && run >= RG_RULE2_CALLS && !rg_changes_files(tool)) {
        bool ok = true;
        for (size_t k = calls.size() - run; k < calls.size(); ++k) {
            if (!short_result(calls[k])) {
                ok = false;
                break;
            }
        }
        if (ok) {
            res.fire = true;
            res.rule = 2;
        }
    }
    if (res.fire) {
        res.count = run;
        res.tool  = tool;
    }
    return res;
}

std::string common_repeat_guard_note(const common_repeat_guard_result & r) {
    return "You have made nearly the same " + r.tool + " call " + std::to_string(r.count) +
           " times in a row. Try a different step or give your answer.";
}
