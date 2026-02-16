from pathlib import Path


def main() -> None:
    p = Path(
        "/mnt/z/Project_sEMG_FMA/.venv/lib/python3.12/site-packages/mamba_ssm/ops/selective_scan_interface.py"
    )
    if not p.exists():
        raise FileNotFoundError(str(p))

    s = p.read_text(encoding="utf-8")
    old_import = "import selective_scan_cuda"
    new_import = (
        "try:\n"
        "    import selective_scan_cuda\n"
        "except Exception:\n"
        "    selective_scan_cuda = None"
    )
    if old_import in s:
        s = s.replace(old_import, new_import)

    old_fn = (
        "def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,\n"
        "                     return_last_state=False):\n"
        "    \"\"\"if return_last_state is True, returns (out, last_state)\n"
        "    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is\n"
        "    not considered in the backward pass.\n"
        "    \"\"\"\n"
        "    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)\n"
    )
    new_fn = (
        "def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,\n"
        "                     return_last_state=False):\n"
        "    \"\"\"if return_last_state is True, returns (out, last_state)\n"
        "    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is\n"
        "    not considered in the backward pass.\n"
        "    \"\"\"\n"
        "    if selective_scan_cuda is None:\n"
        "        return selective_scan_ref(\n"
        "            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,\n"
        "            delta_softplus=delta_softplus, return_last_state=return_last_state\n"
        "        )\n"
        "    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)\n"
    )
    if old_fn in s:
        s = s.replace(old_fn, new_fn)
    else:
        if "if selective_scan_cuda is None:" not in s:
            raise RuntimeError("Failed to locate selective_scan_fn block to patch.")

    p.write_text(s, encoding="utf-8")
    print(f"patched: {p}")


if __name__ == "__main__":
    main()
