#!/usr/bin/env python3
"""Jalankan 12 notebook CIFAR-10 v5 secara berurutan.

Gunakan dari environment Anaconda yang sudah berisi TensorFlow dan Jupyter:
    python run_all_notebooks.py --list
    python run_all_notebooks.py
    python run_all_notebooks.py --start-at "AVG Training.ipynb"

Notebook dijalankan dari direktori tempat skrip ini berada. Output cell
juga ditulis ke notebook asli di root setelah setiap cell selesai. Salinan
notebook asli sebelum run, hasil eksekusi, dan status disimpan di
outputs/cifar10_7fitur_v5/runner/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
except ImportError as exc:
    raise SystemExit(
        "Pustaka untuk menjalankan notebook belum tersedia di Python ini. "
        "Aktifkan environment Anaconda envGPU, lalu jalankan skrip lagi. "
        f"Detail: {exc}"
    ) from exc


NOTEBOOKS = (
    "RGB Training v4.ipynb",
    "RGB Extraction v4.ipynb",
    "AVG Training.ipynb",
    "AVG Extraction.ipynb",
    "NTSC Training.ipynb",
    "NTSC Extraction.ipynb",
    "Feature Engineering Combination v4.ipynb",
    "Feature Engineering Training v4.ipynb",
    "RGB Testing v4.ipynb",
    "AVG Testing.ipynb",
    "NTSC Testing.ipynb",
    "Feature Engineering Testing v4.ipynb",
)
RUN_ID = "cifar10_7fitur_v5"
KERNEL_NAME = "python3"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_HEADER = re.compile(r"Epoch \d+/\d+")
PROGRESS = re.compile(r"^\s*(\d+)/(\d+)\s+.*(?:/step|steps?/s)")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def save_notebook(path: Path, notebook: nbformat.NotebookNode) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, path)


def compact_cell_outputs(cell: nbformat.NotebookNode) -> None:
    """Simpan ringkasan epoch, tanpa ribuan pembaruan progress per batch."""
    compacted = []
    epoch_header = None
    progress_text = None
    progress_total = None
    progress_name = "stdout"

    def flush_progress() -> None:
        nonlocal epoch_header, progress_text, progress_total
        if progress_text is not None:
            if epoch_header and ("accuracy:" in progress_text or "loss:" in progress_text):
                summary = f"{epoch_header} — {progress_text}"
                epoch_header = None
            else:
                summary = progress_text
            compacted.append(nbformat.v4.new_output("stream", name=progress_name, text=summary + "\n"))
        elif epoch_header:
            compacted.append(nbformat.v4.new_output("stream", name="stdout", text=epoch_header + "\n"))
            epoch_header = None
        progress_text = None
        progress_total = None

    for output in cell.get("outputs", []):
        if output.get("output_type") != "stream":
            flush_progress()
            compacted.append(output)
            continue
        original = output.get("text", "")
        if isinstance(original, list):
            original = "".join(original)
        cleaned = ANSI_ESCAPE.sub("", original).replace("\x08", "").strip("\r\n ")
        if EPOCH_HEADER.fullmatch(cleaned):
            flush_progress()
            epoch_header = cleaned
            continue
        match = PROGRESS.match(cleaned)
        if match:
            total = match.group(2)
            if progress_text is not None and total != progress_total:
                flush_progress()
            progress_text = cleaned
            progress_total = total
            progress_name = output.get("name", "stdout")
            continue
        if cleaned:
            compacted.append(output)
    flush_progress()
    cell.outputs = compacted


def heading_before(notebook: nbformat.NotebookNode, cell_index: int) -> str:
    for index in range(cell_index - 1, -1, -1):
        cell = notebook.cells[index]
        if cell.cell_type == "markdown":
            return cell.source.strip().splitlines()[0].lstrip("# ").strip()
    return f"Cell {cell_index + 1}"


class StreamingNotebookClient(NotebookClient):
    """Tampilkan output teks dan simpan output cell pada notebook asli."""

    def __init__(self, notebook, *, mirror_path: Path, **kwargs):
        super().__init__(notebook, **kwargs)
        self.mirror_path = mirror_path

    async def async_execute_cell(
        self, cell, cell_index, execution_count=None, store_history=True
    ):
        if cell.cell_type != "code" or not cell.source.strip():
            return await super().async_execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )
        title = heading_before(self.nb, cell_index)
        print(f"\n  Cell {cell_index + 1}/{len(self.nb.cells)}: {title}", flush=True)
        started = time.monotonic()
        try:
            result = await super().async_execute_cell(
                cell,
                cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )
        except Exception:
            compact_cell_outputs(cell)
            save_notebook(self.mirror_path, self.nb)
            print(
                f"\n  GAGAL setelah {(time.monotonic() - started) / 60:.1f} menit",
                flush=True,
            )
            raise
        compact_cell_outputs(cell)
        save_notebook(self.mirror_path, self.nb)
        print(
            f"\n  Selesai dalam {(time.monotonic() - started) / 60:.1f} menit",
            flush=True,
        )
        return result

    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            content = msg.get("content", {})
            destination = sys.stderr if content.get("name") == "stderr" else sys.stdout
            destination.write(content.get("text", ""))
            destination.flush()
        return super().process_message(msg, cell, cell_index)


def check_notebooks(root: Path) -> None:
    actual = {path.name for path in root.glob("*.ipynb")}
    expected = set(NOTEBOOKS)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Daftar notebook tidak sesuai. Hilang: {missing}; tambahan: {extra}"
        )
    for name in NOTEBOOKS:
        notebook = nbformat.read(root / name, as_version=4)
        nbformat.validate(notebook)
        kernel = notebook.metadata.get("kernelspec", {}).get("name")
        if kernel != KERNEL_NAME:
            raise RuntimeError(
                f"Kernel {name} adalah {kernel!r}, diharapkan {KERNEL_NAME!r}"
            )
    if KERNEL_NAME not in KernelSpecManager().find_kernel_specs():
        raise RuntimeError(
            f"Kernel Jupyter {KERNEL_NAME!r} tidak tersedia di environment ini"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Tampilkan urutan tanpa menjalankan notebook",
    )
    parser.add_argument(
        "--start-at",
        metavar="NAMA_ATAU_NOMOR",
        help="Mulai dari nama file notebook atau nomor urut 1–12",
    )
    return parser.parse_args()


def start_index(value: str | None) -> int:
    if value is None:
        return 0
    if value.isdecimal():
        number = int(value)
        if 1 <= number <= len(NOTEBOOKS):
            return number - 1
    if value in NOTEBOOKS:
        return NOTEBOOKS.index(value)
    raise ValueError(f"--start-at tidak dikenal: {value!r}")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    # Kernel 'python3' memakai 'python' dari PATH; dahulukan interpreter skrip.
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    check_notebooks(root)
    first = start_index(args.start_at)
    for number, name in enumerate(NOTEBOOKS, 1):
        marker = "  <-- mulai" if number - 1 == first else ""
        print(f"{number:2d}. {name}{marker}")
    if args.list:
        return 0

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / "outputs" / RUN_ID / "runner" / run_stamp
    executed_dir = run_dir / "executed"
    original_dir = run_dir / "original"
    executed_dir.mkdir(parents=True, exist_ok=False)
    original_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    status = {
        "run_id": RUN_ID,
        "started_utc": now_utc(),
        "runner_python": sys.executable,
        "kernel_name": KERNEL_NAME,
        "start_at": NOTEBOOKS[first],
        "notebooks": {},
    }
    save_json(status_path, status)
    print(f"\nStatus run: {status_path}", flush=True)

    for number in range(first, len(NOTEBOOKS)):
        name = NOTEBOOKS[number]
        source_path = root / name
        original_path = original_dir / name
        executed_path = executed_dir / name
        shutil.copy2(source_path, original_path)
        entry = {
            "status": "running",
            "source_sha256": sha256_file(source_path),
            "started_utc": now_utc(),
            "executed_notebook": str(executed_path.relative_to(root)),
            "original_backup": str(original_path.relative_to(root)),
        }
        status["notebooks"][name] = entry
        save_json(status_path, status)
        print(
            f"\n{'=' * 72}\n[{number + 1}/{len(NOTEBOOKS)}] {name}\n{'=' * 72}",
            flush=True,
        )
        notebook = nbformat.read(source_path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        client = StreamingNotebookClient(
            notebook,
            mirror_path=source_path,
            timeout=None,
            kernel_name=KERNEL_NAME,
            resources={"metadata": {"path": str(root)}},
            allow_errors=False,
            force_raise_errors=True,
        )
        started = time.monotonic()
        try:
            client.execute()
        except BaseException as exc:
            for cell in client.nb.cells:
                if cell.cell_type == "code":
                    compact_cell_outputs(cell)
            save_notebook(source_path, client.nb)
            save_notebook(executed_path, client.nb)
            entry.update(
                status="interrupted"
                if isinstance(exc, KeyboardInterrupt)
                else "failed",
                finished_utc=now_utc(),
                elapsed_seconds=round(time.monotonic() - started, 1),
                error=f"{type(exc).__name__}: {exc}",
            )
            save_json(status_path, status)
            print(
                f"\nBerhenti pada {name}. Lihat {executed_path} dan {status_path}.",
                file=sys.stderr,
            )
            print(entry["error"], file=sys.stderr)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        save_notebook(source_path, client.nb)
        save_notebook(executed_path, client.nb)
        entry.update(
            status="completed",
            finished_utc=now_utc(),
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        save_json(status_path, status)
        print(
            f"\nSelesai: {name} ({entry['elapsed_seconds'] / 60:.1f} menit)", flush=True
        )

    status["finished_utc"] = now_utc()
    status["status"] = "completed"
    save_json(status_path, status)
    print(f"\nSeluruh notebook selesai. Status: {status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
