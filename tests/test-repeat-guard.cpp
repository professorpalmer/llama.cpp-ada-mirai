// Unit cases for the repetition guard (common/repeat-guard.h): runs that must fire and agent patterns that must not.
#include "repeat-guard.h"

#include <cstdio>
#include <string>
#include <vector>

static common_chat_msg user(const std::string & s) {
    common_chat_msg m; m.role = "user"; m.content = s; return m;
}
static common_chat_msg sys(const std::string & s) {
    common_chat_msg m; m.role = "system"; m.content = s; return m;
}
static common_chat_msg call(const std::string & name, const std::string & args, const std::string & id) {
    common_chat_msg m; m.role = "assistant"; m.tool_calls.push_back({ name, args, id }); return m;
}
static common_chat_msg result(const std::string & text, const std::string & id) {
    common_chat_msg m; m.role = "tool"; m.content = text; m.tool_call_id = id; return m;
}
static common_chat_msg answer(const std::string & s) {
    common_chat_msg m; m.role = "assistant"; m.content = s; return m;
}

static int n_fail = 0;
static void expect(const char * name, const std::vector<common_chat_msg> & msgs, bool fire, int rule = 0) {
    const auto r = common_repeat_guard_check(msgs);
    const bool ok = r.fire == fire && (!fire || rule == 0 || r.rule == rule);
    printf("%-58s fire %d rule %d count %2d tool %-8s %s\n", name, r.fire, r.rule, r.count, r.tool.c_str(), ok ? "OK" : "FAIL");
    if (!ok) {
        n_fail++;
    }
}

static std::vector<common_chat_msg> base() {
    return { sys("You are a coding agent."), user("Fix the slug function and run the tests.") };
}

int main() {
    const std::string long_text(5000, 'x');
    // 1. the G6 loop: echo end1 .. endN with short results
    {
        auto m = base();
        for (int i = 1; i <= 6; ++i) {
            const std::string id = "c" + std::to_string(i);
            m.push_back(call("bash", "{\"command\":\"echo end" + std::to_string(i) + "\"}", id));
            m.push_back(result("end" + std::to_string(i), id));
        }
        expect("echo end1..end6 (rule 1)", m, true, 1);
        auto m5 = base();
        for (int i = 1; i <= 5; ++i) {
            const std::string id = "c" + std::to_string(i);
            m5.push_back(call("bash", "{\"command\":\"echo end" + std::to_string(i) + "\"}", id));
            m5.push_back(result("end" + std::to_string(i), id));
        }
        expect("echo end1..end5 (one short of rule 1)", m5, false);
        auto ma = m;
        ma.push_back(answer("Done."));
        expect("same run, then a text answer (last is not a tool)", ma, false);
        auto mu = m;
        mu.push_back(user("thanks"));
        expect("same run, then a user message", mu, false);
    }
    // 2. the G6 loop start: different trivial commands, short results (rule 2 at 8)
    {
        const char * cmds[] = { "echo ok", "echo stop", "echo waiting", "echo ready", "echo go", "echo done", "echo break",
                                "echo x" };
        auto m = base();
        for (int i = 0; i < 8; ++i) {
            const std::string id = "d" + std::to_string(i);
            m.push_back(call("bash", std::string("{\"command\":\"") + cmds[i] + "\"}", id));
            m.push_back(result(std::string(cmds[i]).substr(5), id));
        }
        expect("8 different trivial echo calls (rule 2)", m, true, 2);
        m.resize(m.size() - 2);
        expect("7 different trivial echo calls", m, false);
        // the G6 shape (capture line 21): a real bash step with a long result, then 8 trivial ones: the long result
        // ends the run, so the 8 short ones still count
        auto g = base();
        g.push_back(call("bash", "{\"command\":\"python -c \\\"print(open('src/textutil.py').read())\\\"\"}", "L0"));
        g.push_back(result(long_text.substr(0, 900), "L0"));
        for (int i = 0; i < 8; ++i) {
            const std::string id = "h" + std::to_string(i);
            g.push_back(call("bash", std::string("{\"command\":\"") + cmds[i] + "\"}", id));
            g.push_back(result(std::string(cmds[i]).substr(5), id));
        }
        expect("long-result bash, then 8 trivial calls (rule 2)", g, true, 2);
        g.resize(g.size() - 2);
        expect("long-result bash, then 7 trivial calls", g, false);
    }
    // 3. negatives: normal agent work
    {
        auto m = base();
        for (int i = 0; i < 10; ++i) {
            const std::string id = "r" + std::to_string(i);
            m.push_back(call("read", "{\"filePath\":\"src/mod" + std::to_string(i) + ".py\"}", id));
            m.push_back(result(long_text, id));
        }
        expect("10 reads of different files, long results", m, false);
    }
    {
        auto m = base();
        for (int i = 0; i < 6; ++i) {
            const std::string e = "e" + std::to_string(i), t = "t" + std::to_string(i);
            m.push_back(call("edit", "{\"filePath\":\"src/textutil.py\",\"oldString\":\"a" + std::to_string(i) + "\",\"newString\":\"b\"}", e));
            m.push_back(result("ok", e));
            m.push_back(call("bash", "{\"command\":\"python -m unittest\"}", t));
            m.push_back(result("FAILED (failures=" + std::to_string(6 - i) + ")", t));
        }
        expect("edit + run the tests, 6 times (tool alternates)", m, false);
    }
    {
        const char * bodies[] = { "def slugify(s): return s.lower()", "import unittest\nclass T(unittest.TestCase): pass",
                                  "# Notes\nThe parser reads one line at a time.", "[tool]\nname = 'demo'\nversion = '0.1'",
                                  "SELECT id, name FROM users WHERE active = 1;", "<html><body><h1>Report</h1></body></html>",
                                  "export function add(a, b) { return a + b; }", "key: value\nlist:\n  - one\n  - two" };
        auto m = base();
        for (int i = 0; i < 8; ++i) {
            const std::string id = "w" + std::to_string(i);
            m.push_back(call("write", std::string("{\"filePath\":\"out/file_") + (char) ('a' + i) + ".txt\",\"content\":\"" + bodies[i] + "\"}", id));
            m.push_back(result("wrote", id));
        }
        expect("8 writes of different files and contents (no rule 2)", m, false);
    }
    {
        auto m = base();
        for (int i = 0; i < 8; ++i) {
            const std::string id = "g" + std::to_string(i);
            m.push_back(call("grep", "{\"pattern\":\"legacy_connect\",\"path\":\"src/" + std::string(1, (char) ('a' + i)) + "\"}", id));
            m.push_back(result(long_text.substr(0, 300), id));
        }
        expect("8 greps with results over 200 characters", m, false);
    }
    {
        auto m = base();
        for (int i = 0; i < 4; ++i) {
            const std::string id = "p" + std::to_string(i);
            m.push_back(call("bash", "{\"command\":\"echo end" + std::to_string(i) + "\"}", id));
            m.push_back(result("end", id));
        }
        m.push_back(user("keep going"));
        for (int i = 4; i < 8; ++i) {
            const std::string id = "p" + std::to_string(i);
            m.push_back(call("bash", "{\"command\":\"echo end" + std::to_string(i) + "\"}", id));
            m.push_back(result("end", id));
        }
        expect("4 + 4 repeats split by a user message", m, false);
    }
    {
        // two calls per turn (parallel calls) never count for rule 1
        auto m = base();
        for (int i = 0; i < 6; ++i) {
            const std::string a = "a" + std::to_string(i), b = "b" + std::to_string(i);
            common_chat_msg c; c.role = "assistant";
            c.tool_calls.push_back({ "read", "{\"filePath\":\"x" + std::to_string(i) + "\"}", a });
            c.tool_calls.push_back({ "glob", "{\"pattern\":\"*.py\"}", b });
            m.push_back(c);
            m.push_back(result("ok", a));
            m.push_back(result("ok", b));
        }
        expect("parallel read + glob per turn, short results", m, false);
    }
    {
        // one turn with 8 parallel reads of small files (OpenCode / Claude Code do this): one step, not a loop
        auto m = base();
        common_chat_msg c; c.role = "assistant";
        for (int i = 0; i < 8; ++i) {
            c.tool_calls.push_back({ "read", "{\"filePath\":\"src/m" + std::to_string(i) + ".py\"}", "q" + std::to_string(i) });
        }
        m.push_back(c);
        for (int i = 0; i < 8; ++i) {
            m.push_back(result("x = 1", "q" + std::to_string(i)));
        }
        expect("one turn with 8 parallel read calls, short results", m, false);
        // the same 8 reads over 2 turns of 4: still under 4 turns
        auto m2 = base();
        for (int t = 0; t < 2; ++t) {
            common_chat_msg c2; c2.role = "assistant";
            for (int i = 0; i < 4; ++i) {
                c2.tool_calls.push_back({ "read", "{\"filePath\":\"src/n" + std::to_string(t * 4 + i) + ".py\"}", "s" + std::to_string(t * 4 + i) });
            }
            m2.push_back(c2);
            for (int i = 0; i < 4; ++i) {
                m2.push_back(result("y = 2", "s" + std::to_string(t * 4 + i)));
            }
        }
        expect("8 parallel reads over 2 turns, short results", m2, false);
    }
    printf("repeat-guard tests: %s (%d failed)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
