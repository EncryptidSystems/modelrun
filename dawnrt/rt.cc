// Dawn runtime. Weights stay in their buffers. A run is one command buffer:
// copies, then compute passes, then one submit. The old host wrote the same
// weight into a single binding on every token. This one does not.
#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include <windows.h>

static WGPUStringView Sv(const char* s) {
  WGPUStringView v = WGPU_STRING_VIEW_INIT;
  if (!s) return v;
  v.data = s;
  v.length = std::strlen(s);
  return v;
}

static WGPUStringView Sv(const std::string& s) {
  WGPUStringView v = WGPU_STRING_VIEW_INIT;
  v.data = s.data();
  v.length = s.size();
  return v;
}

static std::string ToStr(WGPUStringView v) {
  if (!v.data || v.length == 0) return {};
  size_t n = v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length;
  return std::string(v.data, n);
}

struct Buf {
  WGPUBuffer b = nullptr;
  uint64_t size = 0;
  std::string name;
};

struct Pipe {
  WGPUComputePipeline pipe = nullptr;
  WGPUPipelineLayout layout = nullptr;
  WGPUBindGroupLayout bgl = nullptr;
  std::string name;
  std::vector<std::string> keys;
  int nbind = 0;
};

struct Group {
  WGPUBindGroup g = nullptr;
  std::string name;
  uint32_t pipe = 0;
};

struct Dev {
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  std::string err;
  std::vector<WGPUShaderModule> shaders;
  std::map<std::string, uint32_t> shaderID;
  std::vector<Buf> bufs;
  std::map<std::string, uint32_t> bufID;
  std::vector<Pipe> pipes;
  std::map<std::string, uint32_t> pipeID;
  std::vector<Group> groups;
  std::map<std::string, uint32_t> groupID;
};

static Dev g;

static void OnUncaptured(WGPUDevice const*, WGPUErrorType, WGPUStringView message, void*, void*) {
  g.err += "dawn: " + ToStr(message) + "\n";
}

static void Pump(bool* done) {
  for (int i = 0; i < 20000 && !*done; ++i) {
    wgpuInstanceProcessEvents(g.instance);
    if (!*done) Sleep(1);
  }
}

static bool Open(std::string* err) {
  WGPUInstanceDescriptor id = WGPU_INSTANCE_DESCRIPTOR_INIT;
  g.instance = wgpuCreateInstance(&id);
  if (!g.instance) {
    *err = "dawn: wgpuCreateInstance failed\n";
    return false;
  }
  struct Ad {
    WGPUAdapter adapter = nullptr;
    bool done = false;
    std::string msg;
  } ad;
  WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
  opts.featureLevel = WGPUFeatureLevel_Core;
  opts.powerPreference = WGPUPowerPreference_HighPerformance;
  WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  acb.mode = WGPUCallbackMode_AllowSpontaneous;
  acb.userdata1 = &ad;
  acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* u1, void*) {
    Ad& a = *static_cast<Ad*>(u1);
    if (status == WGPURequestAdapterStatus_Success) a.adapter = adapter;
    else a.msg = ToStr(message);
    a.done = true;
  };
  wgpuInstanceRequestAdapter(g.instance, &opts, acb);
  if (!ad.done) Pump(&ad.done);
  if (!ad.adapter) {
    *err = "dawn: no adapter" + (ad.msg.empty() ? "" : ": " + ad.msg) + "\n";
    return false;
  }
  g.adapter = ad.adapter;
  WGPULimits limits = WGPU_LIMITS_INIT;
  bool have = wgpuAdapterGetLimits(g.adapter, &limits) == WGPUStatus_Success;
  struct Dd {
    WGPUDevice device = nullptr;
    bool done = false;
    std::string msg;
  } dd;
  WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
  desc.label = Sv("ModelRun Dawn");
  desc.requiredLimits = have ? &limits : nullptr;
  desc.uncapturedErrorCallbackInfo.callback = OnUncaptured;
  desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  desc.deviceLostCallbackInfo.callback = [](WGPUDevice const*, WGPUDeviceLostReason, WGPUStringView, void*, void*) {};
  WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  dcb.mode = WGPUCallbackMode_AllowSpontaneous;
  dcb.userdata1 = &dd;
  dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* u1, void*) {
    Dd& d = *static_cast<Dd*>(u1);
    if (status == WGPURequestDeviceStatus_Success) d.device = device;
    else d.msg = ToStr(message);
    d.done = true;
  };
  wgpuAdapterRequestDevice(g.adapter, &desc, dcb);
  if (!dd.done) Pump(&dd.done);
  if (!dd.device && have) {
    dd = Dd();
    desc.requiredLimits = nullptr;
    wgpuAdapterRequestDevice(g.adapter, &desc, dcb);
    if (!dd.done) Pump(&dd.done);
  }
  if (!dd.device) {
    *err = "dawn: no device" + (dd.msg.empty() ? "" : ": " + dd.msg) + "\n";
    return false;
  }
  g.device = dd.device;
  g.queue = wgpuDeviceGetQueue(g.device);
  return true;
}

struct R {
  const uint8_t* p;
  size_t n;
  size_t o = 0;
  bool ok = true;
  bool need(size_t k) {
    if (!ok || o + k > n) {
      ok = false;
      return false;
    }
    return true;
  }
  uint32_t u32() {
    if (!need(4)) return 0;
    uint32_t v;
    std::memcpy(&v, p + o, 4);
    o += 4;
    return v;
  }
  uint64_t u64() {
    if (!need(8)) return 0;
    uint64_t v;
    std::memcpy(&v, p + o, 8);
    o += 8;
    return v;
  }
  uint8_t u8() {
    if (!need(1)) return 0;
    return p[o++];
  }
  std::string str() {
    uint32_t n = u32();
    if (!need(n)) return {};
    std::string s(reinterpret_cast<const char*>(p + o), n);
    o += n;
    return s;
  }
  const uint8_t* bytes(uint64_t n) {
    if (!need(static_cast<size_t>(n))) return nullptr;
    const uint8_t* b = p + o;
    o += static_cast<size_t>(n);
    return b;
  }
};

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

static void reply(uint32_t rc, const std::string& body) {
  uint32_t n = static_cast<uint32_t>(body.size());
  wr(&rc, 4);
  wr(&n, 4);
  if (n) wr(body.data(), n);
}

static void replyID(uint32_t id) {
  std::string b(4, 0);
  std::memcpy(b.data(), &id, 4);
  reply(0, b);
}

static uint64_t RoundUp(uint64_t a, uint64_t n) { return (n + a - 1) / a * a; }

static bool TakeErr(std::string* err) {
  wgpuInstanceProcessEvents(g.instance);
  if (g.err.empty()) return true;
  *err = g.err;
  g.err.clear();
  return false;
}

static bool DoShader(R& r, std::string* err) {
  std::string name = r.str();
  uint32_t n = r.u32();
  const uint8_t* src = r.bytes(n);
  if (!r.ok || !src) {
    *err = "shader: short read\n";
    return false;
  }
  std::string code(reinterpret_cast<const char*>(src), n);
  WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
  wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl.code = Sv(code);
  WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  md.nextInChain = &wgsl.chain;
  md.label = Sv(name);
  g.err.clear();
  WGPUShaderModule mod = wgpuDeviceCreateShaderModule(g.device, &md);
  if (!mod) {
    *err = "shader: module was not created\n";
    return false;
  }
  struct Ctx {
    std::string text;
    int errors = 0;
    bool done = false;
  } ctx;
  WGPUCompilationInfoCallbackInfo cb = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.userdata1 = &ctx;
  cb.callback = [](WGPUCompilationInfoRequestStatus, WGPUCompilationInfo const* info, void* u1, void*) {
    Ctx& c = *static_cast<Ctx*>(u1);
    for (size_t i = 0; info && i < info->messageCount; ++i) {
      const WGPUCompilationMessage& m = info->messages[i];
      if (m.type == WGPUCompilationMessageType_Error) {
        ++c.errors;
        c.text += "error: " + ToStr(m.message) + "\n";
      }
    }
    c.done = true;
  };
  wgpuShaderModuleGetCompilationInfo(mod, cb);
  if (!ctx.done) Pump(&ctx.done);
  if (ctx.errors || !TakeErr(err)) {
    if (!ctx.text.empty()) *err += ctx.text;
    wgpuShaderModuleRelease(mod);
    return false;
  }
  g.shaders.push_back(mod);
  uint32_t id = static_cast<uint32_t>(g.shaders.size());
  g.shaderID[name] = id;
  replyID(id);
  return true;
}

static bool DoBuf(R& r, std::string* err) {
  std::string name = r.str();
  uint64_t size = r.u64();
  uint64_t dn = r.u64();
  const uint8_t* data = dn ? r.bytes(dn) : nullptr;
  if (!r.ok || (dn && !data)) {
    *err = "buf: short read\n";
    return false;
  }
  if (size < dn) size = dn;
  size = RoundUp(4, size < 16 ? 16 : size);
  auto it = g.bufID.find(name);
  WGPUBuffer buf = nullptr;
  uint32_t id = 0;
  if (it == g.bufID.end()) {
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.label = Sv(name);
    bd.size = size;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    buf = wgpuDeviceCreateBuffer(g.device, &bd);
    if (!buf || !TakeErr(err)) {
      if (err->empty()) *err = "buf: create failed\n";
      return false;
    }
    g.bufs.push_back(Buf{buf, size, name});
    id = static_cast<uint32_t>(g.bufs.size());
    g.bufID[name] = id;
  } else {
    id = it->second;
    buf = g.bufs[id - 1].b;
    if (size > g.bufs[id - 1].size) {
      *err = "buf: " + name + " is smaller than the write\n";
      return false;
    }
  }
  if (dn) {
    uint64_t padded = RoundUp(4, dn);
    std::vector<uint8_t> tmp(static_cast<size_t>(padded), 0);
    std::memcpy(tmp.data(), data, static_cast<size_t>(dn));
    wgpuQueueWriteBuffer(g.queue, buf, 0, tmp.data(), tmp.size());
  }
  replyID(id);
  return true;
}

static bool DoWrite(R& r, std::string* err) {
  std::string name = r.str();
  uint32_t off = r.u32();
  uint32_t dn = r.u32();
  const uint8_t* data = dn ? r.bytes(dn) : nullptr;
  if (!r.ok || (dn && !data)) {
    *err = "write: short read\n";
    return false;
  }
  auto it = g.bufID.find(name);
  if (it == g.bufID.end()) {
    *err = "write: no buffer " + name + "\n";
    return false;
  }
  Buf& b = g.bufs[it->second - 1];
  uint32_t padded = static_cast<uint32_t>(RoundUp(4, dn));
  if (static_cast<uint64_t>(off) + padded > b.size) {
    *err = "write: past the end of " + name + "\n";
    return false;
  }
  std::vector<uint8_t> tmp(padded, 0);
  if (dn) std::memcpy(tmp.data(), data, dn);
  wgpuQueueWriteBuffer(g.queue, b.b, off, tmp.data(), tmp.size());
  replyID(it->second);
  return true;
}

static bool DoPipe(R& r, std::string* err) {
  std::string name = r.str();
  std::string shader = r.str();
  std::string entry = r.str();
  std::string over = r.str();
  uint32_t nb = r.u32();
  std::vector<uint8_t> kinds(nb);
  for (uint32_t i = 0; i < nb; ++i) kinds[i] = r.u8();
  if (!r.ok) {
    *err = "pipe: short read\n";
    return false;
  }
  auto sit = g.shaderID.find(shader);
  if (sit == g.shaderID.end()) {
    *err = "pipe: no shader " + shader + "\n";
    return false;
  }
  std::vector<WGPUBindGroupLayoutEntry> entries(nb);
  for (uint32_t i = 0; i < nb; ++i) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = i;
    e.visibility = WGPUShaderStage_Compute;
    if (kinds[i] == 2) e.buffer.type = WGPUBufferBindingType_Uniform;
    else if (kinds[i] == 1) e.buffer.type = WGPUBufferBindingType_Storage;
    else e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    entries[i] = e;
  }
  WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  ld.entryCount = entries.size();
  ld.entries = entries.data();
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(g.device, &ld);
  WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g.device, &pld);

  std::vector<std::string> keys;
  std::vector<double> vals;
  std::string cur;
  auto flush = [&]() {
    if (cur.empty()) return;
    auto eq = cur.find('=');
    if (eq == std::string::npos) return;
    keys.push_back(cur.substr(0, eq));
    std::string v = cur.substr(eq + 1);
    vals.push_back(v == "true" ? 1.0 : v == "false" ? 0.0 : std::strtod(v.c_str(), nullptr));
    cur.clear();
  };
  for (char c : over) {
    if (c == ' ' || c == '\n' || c == '\t') flush();
    else cur.push_back(c);
  }
  flush();
  std::vector<WGPUConstantEntry> constants(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    constants[i] = WGPU_CONSTANT_ENTRY_INIT;
    constants[i].key = Sv(keys[i]);
    constants[i].value = vals[i];
  }
  WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
  cd.layout = pl;
  cd.label = Sv(name);
  cd.compute.module = g.shaders[sit->second - 1];
  cd.compute.entryPoint = Sv(entry);
  cd.compute.constantCount = constants.size();
  cd.compute.constants = constants.empty() ? nullptr : constants.data();
  g.err.clear();
  WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g.device, &cd);
  if (!pipe || !TakeErr(err)) {
    if (err->empty()) *err = "pipe: " + name + " was not created\n";
    if (pipe) wgpuComputePipelineRelease(pipe);
    wgpuPipelineLayoutRelease(pl);
    wgpuBindGroupLayoutRelease(bgl);
    return false;
  }
  Pipe p;
  p.pipe = pipe;
  p.layout = pl;
  p.bgl = bgl;
  p.name = name;
  p.keys = std::move(keys);
  p.nbind = static_cast<int>(nb);
  g.pipes.push_back(std::move(p));
  uint32_t id = static_cast<uint32_t>(g.pipes.size());
  g.pipeID[name] = id;
  replyID(id);
  return true;
}

static bool DoGroup(R& r, std::string* err) {
  std::string name = r.str();
  std::string pipe = r.str();
  uint32_t n = r.u32();
  struct Slot {
    std::string buf;
    uint32_t off;
    uint32_t size;
  };
  std::vector<Slot> slots(n);
  for (uint32_t i = 0; i < n; ++i) {
    slots[i].buf = r.str();
    slots[i].off = r.u32();
    slots[i].size = r.u32();
  }
  if (!r.ok) {
    *err = "group: short read\n";
    return false;
  }
  auto pit = g.pipeID.find(pipe);
  if (pit == g.pipeID.end()) {
    *err = "group: no pipe " + pipe + "\n";
    return false;
  }
  Pipe& p = g.pipes[pit->second - 1];
  if (static_cast<int>(n) != p.nbind) {
    *err = "group: " + name + " has " + std::to_string(n) + " bindings, pipe wants " + std::to_string(p.nbind) + "\n";
    return false;
  }
  std::vector<WGPUBindGroupEntry> entries(n);
  for (uint32_t i = 0; i < n; ++i) {
    auto bit = g.bufID.find(slots[i].buf);
    if (bit == g.bufID.end()) {
      *err = "group: no buffer " + slots[i].buf + "\n";
      return false;
    }
    Buf& b = g.bufs[bit->second - 1];
    uint64_t off = slots[i].off;
    uint64_t sz = slots[i].size ? slots[i].size : (b.size - off);
    if (off + sz > b.size) {
      *err = "group: " + slots[i].buf + " binding is past the buffer\n";
      return false;
    }
    WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
    e.binding = i;
    e.buffer = b.b;
    e.offset = off;
    e.size = sz;
    entries[i] = e;
  }
  WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bg.layout = p.bgl;
  bg.entryCount = entries.size();
  bg.entries = entries.data();
  bg.label = Sv(name);
  g.err.clear();
  WGPUBindGroup group = wgpuDeviceCreateBindGroup(g.device, &bg);
  if (!group || !TakeErr(err)) {
    if (err->empty()) *err = "group: " + name + " was not created\n";
    return false;
  }
  g.groups.push_back(Group{group, name, pit->second});
  uint32_t id = static_cast<uint32_t>(g.groups.size());
  g.groupID[name] = id;
  replyID(id);
  return true;
}

static bool DoRun(R& r, std::string* err) {
  uint32_t nops = r.u32();
  struct Op {
    uint8_t tag;
    uint32_t a, b, c, d, e;
  };
  std::vector<Op> ops(nops);
  for (uint32_t i = 0; i < nops; ++i) {
    ops[i].tag = r.u8();
    if (ops[i].tag == 1) {
      ops[i].a = r.u32();
      ops[i].b = r.u32();
      ops[i].c = r.u32();
      ops[i].d = r.u32();
      ops[i].e = r.u32();
    } else if (ops[i].tag == 2) {
      ops[i].a = r.u32();
      ops[i].b = r.u32();
      ops[i].c = r.u32();
      ops[i].d = r.u32();
      ops[i].e = r.u32();
    } else {
      *err = "run: bad op\n";
      return false;
    }
  }
  uint32_t nget = r.u32();
  struct Get {
    uint32_t buf, off, n;
  };
  std::vector<Get> gets(nget);
  for (uint32_t i = 0; i < nget; ++i) {
    gets[i].buf = r.u32();
    gets[i].off = r.u32();
    gets[i].n = r.u32();
  }
  if (!r.ok) {
    *err = "run: short read\n";
    return false;
  }
  g.err.clear();
  WGPUCommandEncoderDescriptor ed = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, &ed);
  WGPUComputePassEncoder pass = nullptr;
  auto endPass = [&]() {
    if (!pass) return;
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    pass = nullptr;
  };
  for (const Op& op : ops) {
    if (op.tag == 2) {
      endPass();
      if (op.a == 0 || op.a > g.bufs.size() || op.c == 0 || op.c > g.bufs.size()) {
        *err = "run: copy buffer id\n";
        wgpuCommandEncoderRelease(enc);
        return false;
      }
      wgpuCommandEncoderCopyBufferToBuffer(enc, g.bufs[op.a - 1].b, op.b, g.bufs[op.c - 1].b, op.d, op.e);
      continue;
    }
    if (op.a == 0 || op.a > g.pipes.size() || op.b == 0 || op.b > g.groups.size()) {
      *err = "run: dispatch id\n";
      endPass();
      wgpuCommandEncoderRelease(enc);
      return false;
    }
    if (!pass) pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, g.pipes[op.a - 1].pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, g.groups[op.b - 1].g, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, op.c ? op.c : 1, op.d ? op.d : 1, op.e ? op.e : 1);
  }
  endPass();
  WGPUCommandBufferDescriptor cd = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cd);
  wgpuQueueSubmit(g.queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);
  struct Ctx {
    bool done = false;
  } ctx;
  WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.userdata1 = &ctx;
  cb.callback = [](WGPUQueueWorkDoneStatus, void* u1, void*) { static_cast<Ctx*>(u1)->done = true; };
  wgpuQueueOnSubmittedWorkDone(g.queue, cb);
  if (!ctx.done) Pump(&ctx.done);
  if (!TakeErr(err)) return false;

  std::string out;
  for (const Get& gt : gets) {
    if (gt.buf == 0 || gt.buf > g.bufs.size()) {
      *err = "run: get buffer id\n";
      return false;
    }
    uint64_t n = RoundUp(4, gt.n);
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = n;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(g.device, &bd);
    WGPUCommandEncoder e2 = wgpuDeviceCreateCommandEncoder(g.device, &ed);
    wgpuCommandEncoderCopyBufferToBuffer(e2, g.bufs[gt.buf - 1].b, gt.off, staging, 0, n);
    WGPUCommandBuffer c2 = wgpuCommandEncoderFinish(e2, &cd);
    wgpuQueueSubmit(g.queue, 1, &c2);
    wgpuCommandBufferRelease(c2);
    wgpuCommandEncoderRelease(e2);
    Ctx done;
    cb.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(g.queue, cb);
    if (!done.done) Pump(&done.done);
    struct Map {
      bool done = false;
      bool ok = false;
      std::string msg;
    } map;
    WGPUBufferMapCallbackInfo mb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mb.mode = WGPUCallbackMode_AllowSpontaneous;
    mb.userdata1 = &map;
    mb.callback = [](WGPUMapAsyncStatus status, WGPUStringView msg, void* u1, void*) {
      Map& m = *static_cast<Map*>(u1);
      m.ok = status == WGPUMapAsyncStatus_Success;
      m.msg = ToStr(msg);
      m.done = true;
    };
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, static_cast<size_t>(n), mb);
    if (!map.done) Pump(&map.done);
    if (!map.ok) {
      *err = "read: " + (map.msg.empty() ? std::string("map failed") : map.msg) + "\n";
      wgpuBufferDestroy(staging);
      wgpuBufferRelease(staging);
      return false;
    }
    const void* p = wgpuBufferGetConstMappedRange(staging, 0, static_cast<size_t>(n));
    if (!p) {
      *err = "read: no range\n";
      wgpuBufferUnmap(staging);
      wgpuBufferDestroy(staging);
      wgpuBufferRelease(staging);
      return false;
    }
    out.append(static_cast<const char*>(p), static_cast<size_t>(gt.n));
    wgpuBufferUnmap(staging);
    wgpuBufferDestroy(staging);
    wgpuBufferRelease(staging);
  }
  reply(0, out);
  return true;
}

static bool SelfTest(std::string* err) {
  const char* src =
      "@group(0) @binding(0) var<storage, read_write> y: array<u32>;\n"
      "@compute @workgroup_size(1)\n"
      "fn main() { y[0] = y[0] + 1u; }\n";
  std::string payload;
  auto u32 = [&](uint32_t v) {
    char b[4];
    std::memcpy(b, &v, 4);
    payload.append(b, 4);
  };
  auto str = [&](const char* s) {
    u32(static_cast<uint32_t>(std::strlen(s)));
    payload.append(s);
  };
  auto send = [&](const std::string& body) -> std::string {
    uint32_t n = static_cast<uint32_t>(body.size());
    // in-process: call the handler by rewinding through the same functions
    R r{reinterpret_cast<const uint8_t*>(body.data()) + 4, body.size() - 4};
    uint32_t op = 0;
    std::memcpy(&op, body.data(), 4);
    bool ok = false;
    if (op == 1) ok = DoShader(r, err);
    else if (op == 2) ok = DoBuf(r, err);
    else if (op == 3) ok = DoPipe(r, err);
    else if (op == 4) ok = DoGroup(r, err);
    else if (op == 5) ok = DoRun(r, err);
    if (!ok) return {};
    (void)n;
    return "ok";
  };
  (void)send;
  (void)src;
  (void)str;
  (void)u32;
  *err = "selftest is the stdin protocol\n";
  return false;
}

static bool Handle(const std::vector<uint8_t>& msg, std::string* err) {
  if (msg.size() < 4) {
    *err = "short command\n";
    return false;
  }
  R r{msg.data() + 4, msg.size() - 4};
  uint32_t op = 0;
  std::memcpy(&op, msg.data(), 4);
  if (op == 1) return DoShader(r, err);
  if (op == 2) return DoBuf(r, err);
  if (op == 7) return DoWrite(r, err);
  if (op == 3) return DoPipe(r, err);
  if (op == 4) return DoGroup(r, err);
  if (op == 5) return DoRun(r, err);
  *err = "unknown op\n";
  return false;
}

int main(int argc, char** argv) {
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
  std::string err;
  if (!Open(&err)) {
    std::fprintf(stderr, "%s", err.c_str());
    return 1;
  }
  if (argc > 1 && std::strcmp(argv[1], "info") == 0) {
    std::fprintf(stderr, "dawnrt ready\n");
    return 0;
  }
  (void)SelfTest;
  for (;;) {
    uint32_t n = 0;
    if (!rd(&n, 4)) break;
    if (n == 0) continue;
    if (n > 0x80000000u) {
      reply(1, "command too large\n");
      return 1;
    }
    std::vector<uint8_t> msg(n);
    if (!rd(msg.data(), n)) break;
    if (n >= 4 && std::memcmp(msg.data(), "quit", 4) == 0) {
      reply(0, {});
      break;
    }
    std::string e;
    if (!Handle(msg, &e)) reply(1, e.empty() ? "failed\n" : e);
  }
  return 0;
}
