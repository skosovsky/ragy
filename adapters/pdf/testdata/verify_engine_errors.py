"""Actual engine exception taxonomy, independently injected after dependency import."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("ragy_pdf_engine", Path(__file__).resolve().parents[1] / "engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

for error, expected in [
    (engine.PdfReadError("private source bytes"), "invalid_pdf"),
    (engine.PDFSyntaxError("private syntax"), "invalid_pdf"),
    (engine.PSEOF("private EOF"), "invalid_pdf"),
    (engine.PSSyntaxError("private tokens"), "invalid_pdf"),
    (RuntimeError("private engine bug"), "engine_internal_error"),
    (ValueError("private dependency bug"), "engine_internal_error"),
    (IndexError("private implementation bug"), "engine_internal_error"),
    (engine._UnsupportedGeometry("private geometry"), "unsupported_geometry"),
    (NotImplementedError("private dependency bug"), "engine_internal_error"),
    (engine._LimitExceeded("private limit"), "limit_exceeded"),
    (OverflowError("private dependency bug"), "engine_internal_error"),
]:
    # Arrange: imports are real, only the parse boundary is replaced.
    def fail(*args):
        raise error
    engine.parse = fail
    sys.argv = ["engine.py", "10", "100", "100", "10"]
    sys.stdin = io.TextIOWrapper(io.BytesIO(b"authorized private bytes"))
    captured = io.StringIO()
    # Act.
    with contextlib.redirect_stdout(captured):
        engine.main()
    # Assert.
    output = json.loads(captured.getvalue())
    assert output == dict(page_count=0, diagnostics=[], pages=[], error=expected), output
    assert "private" not in captured.getvalue()
print("actual imported PDF engine: 11 sanitized exception classes PASS")
