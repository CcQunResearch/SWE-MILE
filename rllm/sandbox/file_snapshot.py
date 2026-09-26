"""Bounded, byte-counted file snapshots shared by verifier transports."""

DEFAULT_FILE_BYTES = 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024


def validate_max_bytes(value: int) -> int:
    if type(value) is not int or not 0 < value <= MAX_FILE_BYTES:
        raise ValueError(f"max_bytes must be an integer in [1, {MAX_FILE_BYTES}]")
    return value


FILE_SNAPSHOT_SCRIPT = '''import base64,hashlib,json,os,stat,sys
result = {}
limit = int(sys.argv[2])
if not 0 < limit <= 8388608:
    raise ValueError("invalid file snapshot byte limit")
for path in json.loads(base64.b64decode(sys.argv[1])):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                result[path] = {"state": "invalid", "error": "not_regular_file"}
                continue
            raw = source.read(limit + 1)
        if len(raw) > limit:
            result[path] = {"state": "invalid", "error": "file_too_large", "size": max(info.st_size, len(raw)), "max_bytes": limit}
        else:
            result[path] = {"state": "present", "content": raw.decode("utf-8"), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    except FileNotFoundError:
        result[path] = {"state": "missing"}
    except UnicodeDecodeError:
        result[path] = {"state": "invalid", "error": "invalid_utf8"}
    except Exception as exc:
        result[path] = {"state": "unreadable", "error": type(exc).__name__}
print("__RLLM_FILE_SNAPSHOT__" + json.dumps(result))
'''
