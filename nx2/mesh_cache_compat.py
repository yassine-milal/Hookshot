"""Preserve raw SPIR-V caches across renderer-only mesh fallback changes.

Only the exact retention/availability edits below are normalized. Any other
translator edit still changes the cache stamp normally. The cache continues
to store original task/mesh SPIR-V, not fallback-generated modules.
"""
def raw_shader_source(data):
    edits = [
        (b'\n            if (p->mesh && dev.mesh_fallback) st.spirv = std::move(spirv);', b''),
        (b'\n            if (dev.mesh_fallback) st.spirv = spirv;', b''),
        (b'\n        if (dev.mesh_fallback) ms.spirv = mspirv;', b''),
        (b'\n            if (dev.mesh_fallback) f.spirv = fspirv;', b''),
        (b'if (!dev.mesh_shader && !dev.mesh_fallback) return nullptr;', b'if (!dev.mesh_shader) return nullptr;'),
    ]
    if not all(data.count(new) == 1 for new, old in edits):
        return data
    for new, old in edits:
        data = data.replace(new, old, 1)
    return data
