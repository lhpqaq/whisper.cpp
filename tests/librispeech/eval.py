import os
import glob
import sys
import re
import jiwer
from normalizers import EnglishTextNormalizer

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u0080-\u009f]")

def _read_text_lossy(path: str) -> tuple[str, bool]:
    """Read a text file that should be UTF-8, but may contain stray bytes.

    Some hypothesis files can end up with non-UTF-8 bytes (e.g. from previous
    tool output). For evaluation we prefer a best-effort decode over crashing.
    """
    raw = open(path, 'rb').read()
    had_decode_error = False
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        had_decode_error = True
        text = raw.decode('utf-8', errors='replace')

    # Remove replacement chars and control codes that can pollute normalization.
    cleaned = text.replace('\ufffd', '')
    cleaned = _CONTROL_CHARS_RE.sub(' ', cleaned)
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
    cleaned = cleaned.strip()

    # Flag sanitization if we had to modify the payload.
    had_sanitize = had_decode_error or (cleaned != text.strip())
    return cleaned, had_sanitize

def get_reference():
    ref = {}
    for path in glob.glob('LibriSpeech/*/*/*/*.trans.txt'):
        # Reference files are expected to be ASCII/UTF-8.
        with open(path, 'r', encoding='utf-8', errors='strict') as fp:
            for line in fp:
                code, text = line.strip().split(" ", maxsplit=1)
                ref[code] = text
    return ref

def get_hypothesis():
    hyp = {}
    for path in glob.glob('LibriSpeech/*/*/*/*.flac.txt'):
        text, sanitized = _read_text_lossy(path)
        if sanitized:
            print(f"warning: sanitized non-utf8/control chars in {path}", file=sys.stderr)
        code = os.path.basename(path).replace('.flac.txt', '')
        hyp[code] = text
    return hyp

def get_codes():
    codes = []
    for path in glob.glob('LibriSpeech/*/*/*/*.flac'):
        codes.append(os.path.basename(path).replace('.flac', ''))
    return sorted(codes)

def main():
    normalizer = EnglishTextNormalizer()

    ref_orig = get_reference()
    hyp_orig = get_hypothesis()

    ref_clean = []
    hyp_clean = []

    for code in get_codes():
        ref_clean.append(normalizer(ref_orig[code]))
        hyp_clean.append(normalizer(hyp_orig[code]))

    wer = jiwer.wer(ref_clean, hyp_clean)
    print(f"WER: {wer * 100:.6f}%")

if __name__ == '__main__':
    main()
