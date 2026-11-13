// Pascal fallback (p4000, dev.mesh_port). The SPIR-V converter/builder are shared with our Lichii layer.
// Original stages stay in descriptor order; only the host pipeline stages change.
struct MeshFallback {
    VkShaderModule vertex = VK_NULL_HANDLE, fragment = VK_NULL_HANDLE;
    VkShaderModule mesh = VK_NULL_HANDLE, task = VK_NULL_HANDLE;
    VkPipeline mesh_pipeline = VK_NULL_HANDLE, task_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout buffers = VK_NULL_HANDLE;
    uint32_t max_v = 0, max_p = 0, payload = 1, nv = 0, np = 0, per_task = 1;
    bool ok = false;
    // (the compute pipelines: made by the precompile's workers or at the first draw, whichever comes first; once)
    std::mutex build;
    std::atomic<int> pipelines{0};   // (0 not made, 1 made, 2 failed)
};
std::unordered_map<const Program*, MeshFallback> mesh_fallbacks;   // (nodes: a reference stays good while others come and go)
std::recursive_mutex mesh_fallback_mu;
std::mutex mesh_port_layout_mu;   // (ensure_layout of our programs: one at a time, a draw and the precompile's workers alike)
// (what became of the programs, for one summary line (mesh_port_summary) instead of a line a program at every start)
std::atomic<uint32_t> mesh_port_converted{0}, mesh_port_unavailable{0}, mesh_port_pipelines_failed{0};
VkResult cached_compute(const VkComputePipelineCreateInfo* ci, VkPipeline* pl);   // (vk_draw.cpp, below: the app's pipeline cache, the driver's faults caught)
extern std::atomic<int64_t> pcache_built_since_save;   // (vk_draw.cpp, below: the pipeline cache has something to save)
MeshFallback& mesh_fallback_get(const Program& p) {
    std::lock_guard lk(mesh_fallback_mu);
    // (always there for a program whose layout was made, which every caller checks first; an empty one, never an exception,
    // if a caller did not: its null modules and pipelines leave the draw out)
    static MeshFallback none;
    const auto it = mesh_fallbacks.find(&p);
    return it != mesh_fallbacks.end() ? it->second : none;
}

void mesh_fallback_forget(const Program& p) {
    std::lock_guard lk(mesh_fallback_mu);
    auto it = mesh_fallbacks.find(&p);
    if (it == mesh_fallbacks.end()) return;
    MeshFallback& f = it->second;
    for (VkPipeline x : {f.mesh_pipeline, f.task_pipeline})
        if (x) vkDestroyPipeline(dev.device, x, nullptr);
    for (VkShaderModule x : {f.vertex, f.fragment, f.mesh, f.task})
        if (x) vkDestroyShaderModule(dev.device, x, nullptr);
    if (f.buffers) vkDestroyDescriptorSetLayout(dev.device, f.buffers, nullptr);
    mesh_fallbacks.erase(it);
}

bool mesh_fallback_prepare(Program& p) {
    std::lock_guard lk(mesh_fallback_mu);
    auto [it, fresh] = mesh_fallbacks.try_emplace(&p);
    if (!fresh) return it->second.ok;
    MeshFallback& f = it->second;
    const Stage *ms = nullptr, *ts = nullptr, *fs = nullptr;
    for (const Stage& st : p.stages) {
        if (st.vk_stage == VK_SHADER_STAGE_MESH_BIT_EXT) ms = &st;
        if (st.vk_stage == VK_SHADER_STAGE_TASK_BIT_EXT) ts = &st;
        if (st.vk_stage == VK_SHADER_STAGE_FRAGMENT_BIT) fs = &st;
    }
    // (the stages' SPIR-V is only for this rewrite, which is made once a program, whatever comes of it: let go at the end,
    // megabytes for the ~1300 mesh programs otherwise)
    struct Release {
        Program& p;
        ~Release() {
            for (Stage& st : p.stages) std::vector<uint32_t>().swap(st.spirv);
        }
    } release{p};
    const auto fail = [&](const std::string& why) {
        // (the first ones in full, then counted: mesh_port_summary. The program is marked failed by ensure_layout: not tried again)
        if (mesh_port_unavailable.fetch_add(1, std::memory_order_relaxed) < 24)
            logf("vk: mesh fallback %016llx unavailable: %s (its draws are left out)", (unsigned long long)p.key, why.c_str());
        mesh_fallback_forget(p);
        return false;
    };
    if (!ms || ms->spirv.empty()) return fail("no retained mesh SPIR-V");
    spvrw::FragResult frag;
    if (fs) {
        frag = spvrw::FragmentForVertexPipeline(fs->spirv.data(), fs->spirv.size());
        if (!frag.ok) return fail(frag.error);
    }
    spvrw::Options opt;
    opt.set = 1;
    opt.task_pipeline = ts != nullptr;
    opt.debug_slot = 0;
    for (const auto& input : frag.inputs) opt.keep_locations.push_back(input.location);
    const auto mesh_code = MeshSingleDrawSpirv(ms->spirv);
    const auto mesh = spvrw::MeshToCompute(mesh_code.data(), mesh_code.size(), opt);
    if (!mesh.ok) return fail(mesh.error);
    spvrw::Result task;
    if (ts) {
        const auto task_code = MeshSingleDrawSpirv(ts->spirv);
        task = spvrw::TaskToCompute(task_code.data(), task_code.size(), opt);
        if (!task.ok) return fail(task.error);
        task.spirv = MeshTaskFirstSpirv(task.spirv, ts->spirv);
        // Hookshot's task epilogue limits emitted groups to its workgroup size by default.
        const char* cap = getenv("RX_VK_TASK_CAP");
        f.per_task = cap && atoi(cap) > 0 ? (uint32_t)atoi(cap) : task.local_x;
        if (!f.per_task || f.per_task > dev.props.limits.maxComputeWorkGroupCount[0]) return fail("task cap exceeds device limits");
    }
    f.max_v = mesh.max_vertices; f.max_p = mesh.max_primitives;
    f.payload = std::max(1u, ts ? task.payload_words : mesh.payload_words);
    f.nv = (uint32_t)mesh.vertex_locations.size(); f.np = (uint32_t)mesh.prim_locations.size();
    const auto module = [&](const std::vector<uint32_t>& code, VkShaderModule& dst) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = code.size() * 4; ci.pCode = code.data();
        return vkCreateShaderModule(dev.device, &ci, nullptr, &dst) == VK_SUCCESS;
    };
    std::vector<spvb::VsOutput> outputs;
    for (const auto& input : frag.inputs) {
        spvb::VsOutput o;
        o.location = input.location; o.component = input.component; o.count = input.count; o.kind = input.kind;
        auto v = std::find(mesh.vertex_locations.begin(), mesh.vertex_locations.end(), input.location);
        auto q = std::find(mesh.prim_locations.begin(), mesh.prim_locations.end(), input.location);
        if (v != mesh.vertex_locations.end()) o.source = 1, o.k = (uint32_t)(v - mesh.vertex_locations.begin());
        else if (q != mesh.prim_locations.end()) o.source = 2, o.k = (uint32_t)(q - mesh.prim_locations.begin());
        outputs.push_back(o);
    }
    if (!module(mesh.spirv, f.mesh) || (ts && !module(task.spirv, f.task)) ||
        !module(spvb::RealVertexShader(1, f.max_v, f.max_p, f.nv, f.np, outputs), f.vertex) ||
        (fs && !module(frag.spirv, f.fragment))) return fail("vkCreateShaderModule");
    VkDescriptorSetLayoutBinding bindings[8]{};
    for (uint32_t b = 0; b < 8; b++)
        bindings[b] = {b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = 8; li.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &f.buffers) != VK_SUCCESS) return fail("buffer descriptor layout");
    f.ok = true;
    if (mesh_port_converted.fetch_add(1, std::memory_order_relaxed) < 6)   // (a few as samples: ~1300 lines a start otherwise)
        logf("vk: mesh fallback %016llx converted: %u vertices, %u primitives, %u vertex/%u primitive outputs, task cap %u",
             (unsigned long long)p.key, f.max_v, f.max_p, f.nv, f.np, f.per_task);
    return true;
}

/// The program's task and mesh compute pipelines, made once (a failure is kept: not compiled again at every draw).
/// wait false (a draw): false as well while another thread is compiling them, a precompile worker the start went on without;
/// the draw is left out until they are there, as other pipelines still compiling are
bool mesh_fallback_pipelines(Program& p, bool wait = true) {
    MeshFallback& f = mesh_fallback_get(p);
    if (const int done = f.pipelines.load(std::memory_order_acquire)) return done == 1;
    std::unique_lock lk(f.build, std::defer_lock);
    if (wait) lk.lock();
    else if (!lk.try_lock()) return false;
    if (const int done = f.pipelines.load(std::memory_order_acquire)) return done == 1;
    const auto make = [&](VkShaderModule module, VkPipeline& out) {
        if (!module || out) return true;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        ci.layout = p.layout;
        pin_subgroup(ci.stage);
        // (as the official fallback's: through the app's pipeline cache, so the next start has them without the driver's own
        // cache, and a fault in the driver's compiler costs this program, not the game)
        const VkResult result = cached_compute(&ci, &out);
        if (result != VK_SUCCESS) {
            out = VK_NULL_HANDLE;
            if (mesh_port_pipelines_failed.fetch_add(1, std::memory_order_relaxed) < 24)
                logf("vk: mesh fallback %016llx compute pipeline failed: %d (its draws are left out)", (unsigned long long)p.key, (int)result);
        } else {
            pcache_built_since_save++;
        }
        return result == VK_SUCCESS;
    };
    const bool made = make(f.task, f.task_pipeline) && make(f.mesh, f.mesh_pipeline);
    f.pipelines.store(made ? 1 : 2, std::memory_order_release);
    return made;
}

/// One line of what became of our programs so far (after the precompile)
void mesh_port_summary(const char* when) {
    logf("vk: mesh fallback (ours) %s: %u programs converted, %u unavailable, %u with a compute pipeline that failed", when,
         mesh_port_converted.load(), mesh_port_unavailable.load(), mesh_port_pipelines_failed.load());
}
