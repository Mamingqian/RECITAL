"""Trusted standalone supervisor copied into the isolated Python runtime.

Never import this module in the host. Generated source is evaluated only in a
fresh fork after resource limits, descriptor closure, and seccomp installation.
"""
from __future__ import annotations

import ctypes
import errno
import inspect
import json
import os
import resource
import select
import signal
import sys
import time
import traceback
import types
import unittest

sys.path.append(f"/runtime/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
import numpy as np

# The trusted launcher freezes the selected IPC bound before initialization.
MAX_BYTES = int(sys.argv[1])
SECCOMP = ctypes.CDLL("/runtime/deps/libseccomp.so.2", use_errno=True)
LIBC = ctypes.CDLL(None, use_errno=True)
SECCOMP.seccomp_init.argtypes = [ctypes.c_uint32]
SECCOMP.seccomp_init.restype = ctypes.c_void_p
SECCOMP.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
SECCOMP.seccomp_syscall_resolve_name.restype = ctypes.c_int
SECCOMP.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
SECCOMP.seccomp_rule_add.restype = ctypes.c_int
SECCOMP.seccomp_load.argtypes = [ctypes.c_void_p]
SECCOMP.seccomp_load.restype = ctypes.c_int
SECCOMP.seccomp_release.argtypes = [ctypes.c_void_p]


class ArgComparison(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


SECCOMP.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                          ctypes.c_int, ctypes.c_uint,
                                          ctypes.POINTER(ArgComparison)]
SECCOMP.seccomp_rule_add_array.restype = ctypes.c_int


def install_seccomp():
    """Default-deny syscalls, with read-only file opens for stdlib imports."""
    if LIBC.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise RuntimeError("cannot set no_new_privs")
    context = SECCOMP.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("seccomp_init failed")
    allow = 0x7fff0000
    try:
        names = (
            "read write readv writev close fstat newfstatat stat lstat statx lseek "
            "mmap mmap2 mprotect munmap mremap madvise brk rt_sigaction rt_sigprocmask "
            "rt_sigreturn sigaltstack futex futex_time64 sched_yield getrandom "
            "clock_gettime clock_gettime64 gettimeofday time times getrusage "
            "getpid gettid getuid geteuid getgid getegid uname readlink readlinkat "
            "getcwd access faccessat faccessat2 fcntl getdents64 sched_getaffinity "
            "poll ppoll select pselect6 nanosleep clock_nanosleep exit exit_group"
        ).split()
        for name in names:
            number = SECCOMP.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and SECCOMP.seccomp_rule_add(context, allow, number, 0) != 0:
                raise RuntimeError(f"cannot allow syscall {name}")
        # SCMP_CMP_MASKED_EQ=7: no write/create/truncate/append or O_TMPFILE.
        mask = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        mask |= getattr(os, "O_TMPFILE", 0) & ~getattr(os, "O_DIRECTORY", 0)
        for name, argument in (("open", 1), ("openat", 2)):
            number = SECCOMP.seccomp_syscall_resolve_name(name.encode())
            if number >= 0:
                comparison = ArgComparison(argument, 7, mask, 0)
                if SECCOMP.seccomp_rule_add_array(context, allow, number, 1,
                                                   ctypes.byref(comparison)) != 0:
                    raise RuntimeError("cannot restrict file opens")
        if SECCOMP.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        SECCOMP.seccomp_release(context)


def immutable(value):
    if isinstance(value, dict):
        return types.MappingProxyType({key: immutable(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(immutable(child) for child in value)
    return value


def validate_output(role, value, count):
    array = np.asarray(value)
    if role == "library_selector":
        if (array.ndim != 1 or array.dtype.kind not in "iu"
                or len(array) > count or len(np.unique(array)) != len(array)
                or np.any(array < 0) or np.any(array >= count)):
            raise ValueError("selector output must be unique in-range integer pool indices")
        return array.tolist()
    if array.shape != (count,):
        raise ValueError(f"role output must have shape ({count},), got {array.shape}")
    if role == "hard_biological_gate":
        if array.dtype.kind != "b":
            raise ValueError("gate output must be bool (True means violation)")
    else:
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError("utility/penalty output must be finite numeric values")
        if role == "per_sequence_penalty" and np.any(array < 0):
            raise ValueError("penalty output must be nonnegative")
    return array.tolist()


def load_generated(config, references):
    registry = {}

    # Keep the Coder-facing registry API available inside the isolated worker.
    # Coder prompts and unit-test templates are allowed to retrieve the
    # just-registered primitive with ``get_primitive``.  The worker must not
    # import the host registry (which would break isolation and leak unrelated
    # built-ins), so this resolver is deliberately scoped to this one generated
    # module and only exposes the requested registration.
    def get_primitive(role, name):
        try:
            return registry[(role, name)]
        except KeyError as exc:
            raise KeyError(f"unknown isolated primitive {role!r}/{name!r}") from exc

    def register_primitive(role, name):
        if (role, name) != (config["role"], config["name"]):
            raise ValueError("generated registration does not match requested role/name")
        def decorate(cls):
            if registry or not isinstance(cls, type):
                raise ValueError("register exactly one primitive class")
            registry[(role, name)] = cls
            return cls
        return decorate

    namespace = {"__name__": "generated_primitive", "np": np,
                 "register_primitive": register_primitive,
                 "get_primitive": get_primitive, "references": references}
    # These synthetic modules are private to the child, never the host registry.
    package = types.ModuleType("recital")
    package.__path__ = []
    spec = types.ModuleType("recital.spec")
    spec.__path__ = []
    module = types.ModuleType("recital.spec.registry")
    module.register_primitive = register_primitive
    module.get_primitive = get_primitive
    for key, value in {"ROLE_PER_ORACLE_UTILITY": "per_oracle_utility",
                       "ROLE_CROSS_ORACLE_AGG": "cross_oracle_aggregation",
                       "ROLE_PER_SEQ_PENALTY": "per_sequence_penalty",
                       "ROLE_HARD_GATE": "hard_biological_gate",
                       "ROLE_LIBRARY_SELECTOR": "library_selector"}.items():
        setattr(module, key, value)
    sys.modules.update({"recital": package, "recital.spec": spec,
                        "recital.spec.registry": module})
    # dont_inherit: the generated source is compiled without this module's `from __future__ import annotations`, so
    # annotations stay objects and `dataclasses` can resolve them.
    exec(compile(config["code"], "<generated-primitive>", "exec", dont_inherit=True), namespace)
    if len(registry) != 1:
        raise ValueError("source must register exactly one class")
    return next(iter(registry.values())), namespace


def probe_sequences(representation):
    """Three deterministic admission probes inside the task's declared space.

    The whole-alphabet cycle, the homopolymer and the two-letter repeat, drawn
    from the declared alphabet and spanning the declared length range so a
    task whose primitives require full-length sequences can be admitted at all.
    """
    alphabet = representation["alphabet"]
    minimum, maximum = representation["min_length"], representation["max_length"]
    patterns = (alphabet, alphabet[0],
                alphabet[1 % len(alphabet)] + alphabet[2 % len(alphabet)])
    lengths = (minimum, (minimum + maximum) // 2, maximum)
    return [(pattern * (length // len(pattern) + 1))[:length]
            for pattern, length in zip(patterns, lengths)]


def invoke(role, primitive, payload):
    if role == "per_oracle_utility":
        raw = np.asarray(payload["raw"], dtype=float)
        value = primitive(raw, target=immutable(payload["target"]),
                          conditioning=immutable(payload.get("conditioning")))
        count = len(raw)
    elif role == "cross_oracle_aggregation":
        utilities = {key: np.asarray(values, dtype=float)
                     for key, values in payload["utilities"].items()}
        count = len(next(iter(utilities.values())))
        value = primitive(utilities)
    elif role == "library_selector":
        count = len(payload["sequences"])
        measurements = {key: np.asarray(values, dtype=float)
                        for key, values in payload["measurements"].items()}
        value = primitive(list(payload["sequences"]),
                          conditioning=immutable(payload["conditioning"]),
                          measurements=measurements,
                          eligible_indices=immutable(payload["eligible_indices"]),
                          required_counts=immutable(payload["required_counts"]))
        indices = validate_output(role, value, count)
        if any(i not in payload["eligible_indices"] for i in indices):
            raise ValueError("selector returned an ineligible row")
        for group, maximum in payload["required_counts"].items():
            if sum(payload["conditioning"][i] == group for i in indices) > maximum:
                raise ValueError("selector exceeded a group's delivery count")
        return indices
    else:
        count = len(payload["sequences"])
        parameter = inspect.signature(primitive.__call__).parameters.get("conditioning")
        if parameter is not None:
            if parameter.kind not in (inspect.Parameter.KEYWORD_ONLY,
                                       inspect.Parameter.POSITIONAL_OR_KEYWORD):
                raise ValueError("conditioning must be an explicit keyword-capable parameter")
            value = primitive(list(payload["sequences"]),
                              conditioning=immutable(payload.get("conditioning")))
        else:
            value = primitive(list(payload["sequences"]))
    return validate_output(role, value, count)


def child_execution(config, references, message):
    cls, namespace = load_generated(config, references)
    parameters = message.get("parameters", config["parameters"])
    # ``name`` is catalog-owned metadata used when validating the parameter
    # schema.  It must never cross into a generated constructor, even when a
    # malformed caller tries to smuggle it through a worker request.
    if "name" in parameters:
        raise ValueError("framework metadata 'name' is not a constructor parameter")
    inspect.signature(cls).bind(**parameters)
    primitive = cls(**parameters)
    if not callable(primitive):
        raise ValueError("registered primitive instance must be callable")
    if message["op"] == "admit":
        exec(compile(config["unit_test"], "<generated-tests>", "exec", dont_inherit=True), namespace)
        tests = [(name, item) for name, item in namespace.items()
                 if name.startswith("test_") and inspect.isfunction(item)]
        classes = [item for item in namespace.values()
                   if isinstance(item, type) and issubclass(item, unittest.TestCase)
                   and item is not unittest.TestCase]
        if not tests and not classes:
            raise ValueError("unit_test must define executable test_* functions or TestCase")
        for _, test in tests:
            inspect.signature(test).bind()
            test()
        for case in classes:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
            result = unittest.TestResult()
            suite.run(result)
            if not result.testsRun or not result.wasSuccessful() or result.skipped:
                raise ValueError("generated unittest checks did not all pass")
        # Independent generic role check, not a biological qualification test.
        role = config["role"]
        if role == "per_oracle_utility":
            sample = {"raw": [-1.0, 0.0, 1.0], "target": {
                "kind": "maximize", "coordinate": "raw", "units": "score",
                "scale": 1.0, "reference": 0.0}, "conditioning": None}
        elif role == "cross_oracle_aggregation":
            sample = {"utilities": {name: [-1.0, 0.0, 1.0]
                                    for name in config["allowed_oracles"]}}
        elif role == "library_selector":
            schema = config.get("selector_input_schema") or {"conditions": ["all"], "measurements": {
                name: {"shape": [], "sources": [name]} for name in config["allowed_oracles"]}}
            conditions = schema["conditions"]
            vector = np.tile([-1., 0., 1.], len(conditions))
            measurements = {name: (vector.tolist() if not declaration["shape"] else
                                   (vector[:, None] + np.arange(declaration["shape"][0])[None, :] * .1).tolist())
                            for name, declaration in schema["measurements"].items()}
            sample = {"sequences": probe_sequences(config["representation"]) * len(conditions),
                      "conditioning": [condition for condition in conditions for _ in range(3)],
                      "eligible_indices": list(range(3 * len(conditions))),
                      "required_counts": dict.fromkeys(conditions, 2), "measurements": measurements}
        else:
            sample = {"sequences": probe_sequences(config["representation"])}
        output = invoke(role, cls(**parameters), sample)
        return {"sample": output, "test_count": len(tests) + len(classes)}
    if message["op"] != "call":
        raise ValueError("unknown operation")
    return {"output": invoke(config["role"], primitive, message["payload"])}


def generated_location(exc):
    """The innermost frame inside generated source as ' (at <generated-tests> line 4 in test_gate)', or ''."""
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename.startswith("<generated-")]
    if not frames:
        return ""
    return f" (at {frames[-1].filename} line {frames[-1].lineno} in {frames[-1].name})"


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode() + b"\n"


class GeneratedCodeLimit(Exception):
    """The generated code exceeded its CPU, wall-clock or output limit."""


def run_child(config, references, message):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            # Remove supervisor control descriptors before any generated code.
            null_fd = os.open("/dev/null", os.O_RDWR)
            for standard in (0, 1, 2):
                os.dup2(null_fd, standard)
            descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if descriptor_limit == resource.RLIM_INFINITY:
                raise RuntimeError("unbounded inherited descriptor limit")
            os.closerange(3, write_fd)
            os.closerange(write_fd + 1, descriptor_limit)
            limits = config["limits"]
            resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"],) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"],) * 2)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            install_seccomp()
            np.random.seed(0)
            import random
            random.seed(0)
            response = {"ok": True, "result": child_execution(config, references, message)}
        except BaseException as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:1500]}{generated_location(exc)}"}
        try:
            data = encode(response)
            if len(data) > config["limits"]["max_message_bytes"]:
                data = encode({"ok": False, "error": "output exceeds byte limit"})
            while data:
                data = data[os.write(write_fd, data):]
        finally:
            os._exit(0)
    os.close(write_fd)
    limit = config["limits"]["admission_seconds" if message["op"] == "admit" else "wall_seconds"]
    deadline = time.monotonic() + limit
    chunks = bytearray()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([read_fd], [], [], max(0, remaining))[0]:
                raise GeneratedCodeLimit("isolated execution wall timeout")
            part = os.read(read_fd, 65536)
            if not part:
                break
            chunks.extend(part)
            if len(chunks) > config["limits"]["max_message_bytes"]:
                raise GeneratedCodeLimit("isolated output exceeds byte limit")
        # A malicious writer could close its result pipe and spin. Reap boundedly.
        while True:
            finished, status = os.waitpid(pid, os.WNOHANG)
            if finished:
                pid = None
                if status != 0:
                    raise GeneratedCodeLimit("isolated execution terminated (CPU/resource limit or signal)")
                break
            if time.monotonic() >= deadline:
                raise GeneratedCodeLimit("isolated execution wall timeout")
            time.sleep(0.005)
        value = json.loads(chunks)
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise ValueError("invalid isolated response")
        return value
    except GeneratedCodeLimit as exc:
        # The code ran past its CPU, wall-clock or output limit: that is a verdict on the code, which
        # the Coder can repair, not a fault of the sandbox.
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
    except BaseException as exc:
        # A fault in the supervisor's own fork/read/reap loop is infrastructure, not a
        # verdict on the generated code, and must not reach the Coder as its mistake.
        return {"ok": False, "isolation_failure": True,
                "error": f"{type(exc).__name__}: {str(exc)[:1500]}"}
    finally:
        os.close(read_fd)
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)


def main():
    initialization = sys.stdin.buffer.readline(MAX_BYTES + 1)
    if len(initialization) > MAX_BYTES:
        raise ValueError("initialization exceeds byte limit")
    config = json.loads(initialization)
    references = immutable({key: value["data"] for key, value in config.pop("references").items()})
    sys.stdout.buffer.write(encode({"ok": True, "ready": True}))
    sys.stdout.buffer.flush()
    while True:
        line = sys.stdin.buffer.readline(config["limits"]["max_message_bytes"] + 1)
        if not line:
            return
        if len(line) > config["limits"]["max_message_bytes"]:
            return
        message = json.loads(line)
        response = run_child(config, references, message)
        sys.stdout.buffer.write(encode(response))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
