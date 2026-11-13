"""Float atomic adds and 8/16-bit global stores are native where the GPU can, and never touch a view that is not there.

VK_EXT_shader_atomic_float (shaderBufferFloat32AtomicAdd) gives OpAtomicFAddEXT: the compare-exchange loops retried under
contention (two OoT passes 14-37 ms a frame on a Quadro P4000). An earlier version made the global float-add helper for
every program that stores and built an access chain on an F32 view the program did not have: the game aborted at start
(0xC0000409). So: the helper only for programs that add floats, define_body skips a missing view, and the switches are
RX_VK_NATIVE* (the program cache's stamp hashes those).

python -I -B tests/test_native_float_add.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VK = ROOT / "gpu" / "vk"
SPIRV = VK / "eden" / "shader_recompiler" / "backend" / "spirv"


def need(cond, why):
    if not cond:
        raise SystemExit(why)


def main():
    main_cpp = (VK / "vk_main.cpp").read_text(encoding="utf-8")
    shader = (VK / "vk_shader.cpp").read_text(encoding="utf-8")
    ctx = (SPIRV / "spirv_emit_context.cpp").read_text(encoding="utf-8")
    atomic = (SPIRV / "emit_spirv_atomic.cpp").read_text(encoding="utf-8")
    memory = (SPIRV / "emit_spirv_memory.cpp").read_text(encoding="utf-8")
    info = (VK / "eden" / "shader_recompiler" / "ir_opt" / "collect_shader_info_pass.cpp").read_text(encoding="utf-8")
    sirit = (VK / "sirit" / "src" / "instructions" / "atomic.cpp").read_text(encoding="utf-8")

    need("optional(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, dev.atomic_float);" in main_cpp, "the extension is not asked for")
    need("if (dev.atomic_float) *tail = &afloat" in main_cpp, "its features are not queried and enabled")
    need("afloat.shaderBufferFloat32AtomicAdd" in main_cpp and 'getenv("RX_VK_NATIVE_FADD")' in main_cpp,
         "dev.atomic_float_add ignores the feature bit or has no RX_VK_NATIVE switch")
    need("p.support_float32_atomic_add = dev.atomic_float_add;" in shader, "the profile is not told")
    need('getenv("RX_VK_NATIVE_NARROW")' in shader, "the narrow stores have no RX_VK_NATIVE switch")
    need('"RX_VK_NATIVE"' in shader, "the program cache's stamp does not hash RX_VK_NATIVE* switches")
    need("profile().support_float32_atomic_add ? 1u : 0u" in shader, "the stamp does not tell a native-add cache apart")

    need("OpAtomicFAddEXT" in sirit, "sirit has no OpAtomicFAdd")
    need("ctx.profile.support_float32_atomic_add && ctx.profile.support_descriptor_aliasing" in atomic and
         "spv::Capability::AtomicFloat32AddEXT" in atomic and "&StorageDefinitions::F32" in atomic,
         "the storage buffer float add is not native")
    need("info.used_storage_buffer_types |= IR::Type::U32 | IR::Type::F32;" in info, "a float add declares no F32 view")
    # the guard: only programs that add floats get the global helper, and a missing view is skipped, never chained on
    need("if (info.stores_global_memory && info.uses_atomic_f32_add) {" in ctx, "the global float add helper is made for every storing program")
    need("if (!Sirit::ValidId(element_pointer)) return;" in ctx, "define_body builds on a view type that does not exist")
    need("!Sirit::ValidId(ssbos[index].*ssbo_member)" in ctx, "define_body builds on a buffer without that view")
    need("native_add(psb_member(F32[1], target))" in ctx, "the page-table path of the global float add is not native")
    need("const Id word_address{OpBitwiseAnd(U64, addr, Constant(U64, ~u64{3}))};" in ctx,
         "the word compare-exchange fallback looks up the unaligned address")
    need("info.stores_global_memory && profile.support_native_narrow_global_store" in ctx and
         "ctx.write_global_u8_func : ctx.write_global_u16_func" in memory, "8/16-bit global stores are not direct")
    print("native float atomic add and narrow global stores: ok")


if __name__ == "__main__":
    main()
