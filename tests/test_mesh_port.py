"""Our mesh renderer (dev.mesh_port, GPUs without mesh shaders) fails soft.

It replaces the official compute fallback by default on such a GPU, so each of its failures must cost the least:
  - its scratch or argument program cannot be made: the official fallback runs instead, decided before the program
    cache is opened (its stamp differs), and the game starts;
  - a program whose rewrite fails is marked failed once (ensure_layout) and its draws left out, a line for the first ones;
  - a program whose compute pipeline fails is not compiled again, and the compile goes through the app's pipeline cache
    with the driver's faults caught, as the official fallback's;
  - a draw never waits on a compile another thread holds;
  - the GPU's size reports are read before the recording's grown streams are given back.

python -I -B tests/test_mesh_port.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def need(cond, why):
    if not cond:
        raise SystemExit(why)


def main():
    main_cpp = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    replay = (VK / "vk_replay.cpp").read_text(encoding="utf-8")
    draw = (VK / "vk_draw.cpp").read_text(encoding="utf-8")
    prepare = (VK / "vk_mesh_prepare.h").read_text(encoding="utf-8")
    mesh_draw = (VK / "vk_mesh_draw.h").read_text(encoding="utf-8")
    shader = (VK / "vk_shader.cpp").read_text(encoding="utf-8")

    # the official path's flags are remembered, and given back when ours cannot be set up
    need("dev.official_mesh_fallback = dev.mesh_fallback, dev.official_task_mesh_fallback = dev.task_mesh_fallback;" in main_cpp,
         "vk_main.cpp: the official fallback's flags are not kept before ours takes over")
    abandon = block(main_cpp, "void mesh_port_abandon(")
    need("dev.mesh_port = false;" in abandon and "dev.mesh_fallback = dev.official_mesh_fallback" in abandon,
         "mesh_port_abandon does not hand over to the official fallback")
    start = block(replay, "bool replay_start(")
    need("if (!mesh_fallback_init()) return false" not in start, "a failed scratch still stops the game from starting")
    need("if (!mesh_fallback_init()) mesh_port_abandon(" in start, "replay_start does not fall back to the official path")
    # (the stamp is read when the cache is opened, after replay_start: it must still tell the two apart)
    need("(dev.mesh_port ? 8u : 0u)" in block(shader, "uint64_t disk_stamp("), "the program cache's stamp has no bit for our renderer")

    # a failed rewrite: once
    layout = block(draw, "void ensure_layout(")
    need("if (p.layout || p.failed) return;" in layout and "if (!mesh_fallback_prepare(p)) { p.failed = true; return; }" in layout,
         "ensure_layout tries a failed rewrite again")
    # a failed compute pipeline: once, through the cache and the fault guard
    pipelines = block(prepare, "bool mesh_fallback_pipelines(")
    need("cached_compute(&ci, &out)" in pipelines and "vkCreateComputePipelines" not in pipelines,
         "our compute pipelines bypass the pipeline cache and the driver fault guard")
    need("f.pipelines.store(made ? 1 : 2" in pipelines, "a failed compute pipeline is not remembered")
    need("else if (!lk.try_lock()) return false;" in pipelines, "a draw waits on another thread's compile")
    need("mesh_fallback_pipelines(p, false)" in block(draw, "void draw(VkCommandBuffer cmd, NvnState& s, const DrawArgs& d)"),
         "the draw asks for the pipelines waiting")
    need(".at(&p)" not in prepare, "mesh_fallback_get can throw")
    # the stages' SPIR-V is let go after the rewrite
    need("std::vector<uint32_t>().swap(st.spirv)" in block(prepare, "bool mesh_fallback_prepare("),
         "the stages' SPIR-V is kept after the rewrite")

    # the reports are read before the streams they are in can be destroyed
    begin = block(draw, "void draw_begin_recording(")
    need("mesh_feedback_read(r);" in begin, "the mesh reports are not read at the recording's start")
    need(begin.index("mesh_feedback_read(r);") < begin.index("destroy_buffer(r.streams[i].buffer"),
         "the mesh reports are read after the recording's grown streams are destroyed")

    # no line a draw
    need("if (told < 20 && ++told)" in mesh_draw, "a direct draw past the scratch logs at every draw")
    print("our mesh renderer fails soft: ok")


if __name__ == "__main__":
    main()
