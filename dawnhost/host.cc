// Resident host for Dawn. One process, one device. Commands stay on stdin so
// buffers and pipelines survive from one Hodge-Kuramoto dispatch to the next.
// This calls driver::Command, GpuPut, GpuGet, and GpuSeq. It does not change
// the driver or the shaders.
#include "driver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>

static bool rd(void* p, size_t n) {
  auto* b = static_cast<unsigned char*>(p);
  size_t o = 0;
  while (o < n) {
    size_t k = std::fread(b + o, 1, n - o, stdin);
    if (k == 0) return false;
    o += k;
  }
  return true;
}

static bool wr(const void* p, size_t n) {
  auto* b = static_cast<const unsigned char*>(p);
  size_t o = 0;
  while (o < n) {
    size_t k = std::fwrite(b + o, 1, n - o, stdout);
    if (k == 0) return false;
    o += k;
  }
  std::fflush(stdout);
  return true;
}

static void reply(uint32_t rc, const void* p, uint32_t n) {
  if (!wr(&rc, 4) || !wr(&n, 4)) std::exit(2);
  if (n && !wr(p, n)) std::exit(2);
}

static void replyStr(uint32_t rc, const std::string& s) {
  reply(rc, s.data(), static_cast<uint32_t>(s.size()));
}

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> t;
  std::string c;
  for (char ch : s) {
    if (ch == ' ' || ch == '\t') {
      if (!c.empty()) {
        t.push_back(c);
        c.clear();
      }
    } else {
      c.push_back(ch);
    }
  }
  if (!c.empty()) t.push_back(c);
  return t;
}

static bool doSeq(const std::string& cmd) {
  std::vector<std::string> t = split(cmd);
  if (t.size() < 3) {
    replyStr(1, "seq <entry> <groups> [K=N ...]\n");
    return true;
  }
  driver::Step st;
  st.entry = t[1];
  st.groups = static_cast<uint32_t>(std::strtoul(t[2].c_str(), nullptr, 0));
  std::map<std::string, std::string> ov;
  for (size_t i = 3; i < t.size(); ++i) {
    auto eq = t[i].find('=');
    if (eq == std::string::npos || eq == 0) {
      replyStr(1, "bad override " + t[i] + "\n");
      return true;
    }
    ov[t[i].substr(0, eq)] = t[i].substr(eq + 1);
  }
  std::string out;
  bool ok = driver::GpuSeq(std::vector<driver::Step>{st}, ov, &out);
  replyStr(ok ? 0 : 1, out);
  return true;
}

static bool doPut(const std::string& cmd) {
  std::string name = cmd.size() > 4 ? cmd.substr(4) : "";
  while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
  if (name.empty() || name.find(' ') != std::string::npos) {
    replyStr(1, "put <name>\n");
    return true;
  }
  uint32_t nb = 0;
  if (!rd(&nb, 4)) return false;
  if (nb > (64u << 20)) {
    replyStr(1, "put is over 64MiB\n");
    return true;
  }
  uint32_t padded = (nb + 3u) & ~3u;
  std::vector<unsigned char> raw(padded);
  if (nb && !rd(raw.data(), nb)) return false;
  std::vector<uint32_t> words(padded / 4);
  if (!words.empty()) std::memcpy(words.data(), raw.data(), padded);
  bool ok = driver::GpuPut(name, words);
  replyStr(ok ? 0 : 1, ok ? "" : "put failed\n");
  return true;
}

static bool doGet(const std::string& cmd) {
  std::vector<std::string> t = split(cmd);
  if (t.size() < 2) {
    replyStr(1, "get <name> [words]\n");
    return true;
  }
  size_t n = 0;
  if (t.size() >= 3) n = static_cast<size_t>(std::strtoul(t[2].c_str(), nullptr, 0));
  std::vector<uint32_t> w;
  std::string err;
  if (!driver::GpuGet(t[1], &w, &err, n)) {
    replyStr(1, err.empty() ? "get failed\n" : err);
    return true;
  }
  reply(0, w.empty() ? nullptr : w.data(), static_cast<uint32_t>(w.size() * 4));
  return true;
}

int main() {
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  std::string err;
  if (!driver::GpuOpen(&err)) {
    std::fwrite(err.data(), 1, err.size(), stderr);
    return 1;
  }
  for (;;) {
    uint32_t n = 0;
    if (!rd(&n, 4)) return 0;
    if (n > (1u << 20)) return 2;
    std::string cmd(n, '\0');
    if (n && !rd(&cmd[0], n)) return 2;
    while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) cmd.pop_back();
    if (cmd.empty()) {
      replyStr(0, "");
      continue;
    }
    if (cmd == "quit") return 0;
    bool ok = true;
    if (cmd.rfind("put ", 0) == 0) ok = doPut(cmd);
    else if (cmd.rfind("get ", 0) == 0) ok = doGet(cmd);
    else if (cmd.rfind("seq ", 0) == 0) ok = doSeq(cmd);
    else {
      std::string out;
      int rc = driver::Command(cmd, &out);
      replyStr(static_cast<uint32_t>(rc), out);
    }
    if (!ok) return 2;
  }
}
