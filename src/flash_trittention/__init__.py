"""Flash Trittention - Efficient attention mechanisms with two implementations."""

__version__ = "0.1.0"

__all__ = ["flash_trittention", "TrittentionTriton"]

# The kernel implementations pull in Triton/einops, which are unavailable in
# CPU-only environments. Import them lazily so `import flash_trittention` (e.g.
# for `__version__`) succeeds without those deps; the ImportError only surfaces
# when a caller actually accesses the kernel entry points.
_LAZY_ATTRS = {
    "flash_trittention": (".triple_dot_product.flash_trittention", "flash_trittention"),
    "TrittentionTriton": (".dot_product_sum.trittention_triton", "TrittentionTriton"),
}


def __getattr__(name):
    try:
        module_path, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module_path, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
