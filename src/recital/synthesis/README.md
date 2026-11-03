# Run-scoped generated primitives

Integration API (trusted caller, never expose artifact paths to the agent):

```python
from recital.synthesis import ExtensionCatalog, PrimitiveRequest, ReferenceBinder

catalog = ExtensionCatalog(artifact_dir=run_dir / "extensions",
                           allowed_oracles={"optimizer"}, reference_binder=binder,
                           # The task's declared sequence space. Penalties and
                           # gates are tested on probes drawn from it at
                           # admission, so a protein task is probed with protein
                           # sequences of its own declared length.
                           representation={"alphabet": "ACDEFGHIKLMNPQRSTVWY",
                                           "min_length": 80, "max_length": 96})
stage = catalog.stage(PrimitiveRequest(role=role, name=name, code=code,
                                      unit_test=unit_test,
                                      parameters={"weight": 1.0},
                                      reference_bindings=("approved_id",)))
constructor = stage.resolve(role, name)
primitive = constructor(weight=1.0)
# The lifecycle controller screens the complete recipe with this resolver, then:
stage.publish()  # or stage.abort(); abort terminates the worker
constructor = catalog.resolve(role, name)
catalog.close()
```

Only approved names in `allowed_oracles` may occur in aggregation dictionaries;
per-oracle utilities must have an admitted `oracle` or trusted `oracle_name=...`
and are called `(raw, *, target, conditioning=None)`. `oracle_name` is proxy metadata and is
not passed to generated constructors. Other roles are called with utilities
mapping or sequence list. Resolve returns a callable trusted constructor proxy;
its signature describes the JSON-valued parameters admitted for this extension.
Constructor values are checked against the declared parameter schema.

Generated source decorates exactly one class with
`@register_primitive(role="...", name="...")`; `np`, `register_primitive`, and
`references` are supplied inside the isolated namespace. `references` maps
approved binding IDs to immutable data. Generated unit tests define at least one
zero-argument `test_*` function (or a unittest.TestCase). Tests see the source
namespace, `np`, and `references`. Tests are admission implementation checks.

Supported roles: `per_oracle_utility`, `cross_oracle_aggregation`,
`per_sequence_penalty`, `hard_biological_gate`. Output is exactly shape `(N,)`:
finite signed numeric utility/aggregation, finite nonnegative penalty, strict
boolean gate where True means violation. Utility target is an immutable JSON
mapping and conditioning is JSON data; raw values remain raw coordinates.

ReferenceBinder is supplied only by trusted adapters. Its `approve` operation
validates a ReferenceBinding record and a complete nonempty JSON dataset against
its SHA-256, source grant, schema, condition, and sample count; the public agent
request contains binding IDs only, never paths or raw reference data.

`catalog.load(artifact_id)` replays a published artifact through the same admission
checks and returns a staging handle; explicitly publish after the main screen.
Code, test, config, binding records and data, and runtime bundle have reproducible
hashes. Existing immutable artifacts are verified, never overwritten.

For composer integration, pass `PrimitiveRequest(..., oracle="optimizer")` for a
utility. Admission checks this name against the optimizer allowlist and binds it
as trusted proxy metadata; the existing composer can then instantiate with only
authored kwargs. Unbound utility requests instead require explicit trusted
`oracle_name` at construction. Worker calls use `(raw, *, target, conditioning)`;
the target's `.to_dict()` payload becomes an immutable mapping in the child.
Catalog and stage resolvers fall back to existing trusted built-in constructors.

Both catalog and stage expose `capability_declarations()` keyed by `role/name`.
Each descriptor includes a closed `parameters` JSON Schema and `code_sha256`,
`test_sha256`, `config_sha256`, `binding_sha256`, `runtime_sha256`, `artifact_id`.
Pass these declarations to `with_admitted_primitives` for the effective screen
capabilities; publish the stage only after the lifecycle screen accepts the recipe.

`PrimitiveRequest.parameter_schema` optionally admits typed/ranged kwargs,
validated for every constructor call. `parameters` supplies the admission/default
values. Without an explicit schema, each default has a const schema and other
values are rejected. Schemas cannot resolve external references or accept
undeclared parameter names. Constructor signatures expose actual parameter names.

The bundle is copied from the running installed interpreter and NumPy plus their
ELF dependencies. Only that private bundle is mounted read-only; no home,
repository, /etc, /proc, credentials, or host network are exposed. A trusted
supervisor preloads granted references once. Every test, admission and call runs in a
fresh child with closed control descriptors, 1 GiB address-space and 2 s CPU
limits, and default-deny seccomp (including process/network/exec denial and
read-only file opens). Calls have a 5 s wall deadline and admission a 10 s wall
deadline, with 0.5 s host transport/reap grace. JSON messages are capped at 8 MiB;
no pickle is used. Those are defaults; see trusted configuration below.
Missing bwrap/seccomp fails closed, with no parent execution.

Penalties and gates expose the pure context interface
`primitive.evaluate_context(sequences, *, conditioning=None)`. Conditioning is
an aligned list/tuple/array of string labels; it is immutable in the execution
child. The worker passes `conditioning=...` only when the generated `__call__`
explicitly declares that keyword-capable parameter. Sequence-only methods
(including methods with only generic `**kwargs`) receive sequences
only. The composer calls this interface for conditional diversity or gate checks;
there are no task- or DNA-specific flags.

`RuntimeLimits` is trusted run configuration, not an agent request field. The
values above are defaults, not admission policy ceilings. The run protocol chooses
and freezes positive finite limits (integer memory bytes, CPU seconds and message
bytes); replay verifies exactly the same limits. The chosen message bound applies
to initialization, calls, responses, and artifact reads. Complete granted reference
data is preserved; payloads over the configured bound fail instead of truncating.
