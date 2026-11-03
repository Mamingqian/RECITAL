"""Typed recipe admission using isolated run-scoped primitive execution.

Staging does not grant an agent permission. Publish only after the final,
normalized recipe passes the same source, capability, binding and execution
checks that apply during training. The original contract remains immutable.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from recital.capabilities import (
    RecipeCapabilities, validate_recipe_capabilities, with_admitted_primitives,
    primitive_request_problems,
)
from recital.proposal_screen import ProposalScreen
from recital.spec.composer import SourceRolePolicy, materialize_reward_input, validate_spec
from .catalog import ExtensionCatalog, StagedExtension
from .models import PrimitiveRequest, SynthesisError


ROLE_FIELDS = {
    "per_oracle_utility": "oracle_utilities",
    "cross_oracle_aggregation": "aggregator",
    "per_sequence_penalty": "penalties",
    "hard_biological_gate": "gates",
}


@dataclass
class PreparedRecipe:
    spec: dict[str, Any]
    capabilities: RecipeCapabilities
    stage: StagedExtension

    def abort(self) -> None:
        if not self.stage.published:
            self.stage.abort()


class RecipeSynthesis:
    """Shared lifecycle; callers supply task semantics and the actual screen."""

    def __init__(self, catalog: ExtensionCatalog, capabilities: RecipeCapabilities):
        self.catalog = catalog
        self.base_capabilities = capabilities
        for role in capabilities.synthesizable_roles:
            if ROLE_FIELDS.get(role) not in capabilities.allowed_reward_fields:
                raise SynthesisError(
                    f"synthesizable role {role!r} requires editable field {ROLE_FIELDS.get(role)!r}"
                )
        self._reference_capabilities()

    def coder_request(self, request: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
        """Bind actual input schemas, never agent-invented target/reference data."""
        problems = primitive_request_problems(request, self.capabilities, spec)
        if problems:
            raise SynthesisError("; ".join(problems))
        result = copy.deepcopy(request)
        declarations = self.catalog.reference_binder.declarations()
        ids = request.get("reference_bindings") or []
        oracle = request.get("oracle")
        result["runtime_context"] = {
            "target_objective": copy.deepcopy(
                (materialize_reward_input(spec).get("target_objectives") or {}).get(oracle)),
            "approved_references": {identifier: declarations[identifier] for identifier in ids},
        }
        return result

    def _reference_capabilities(self) -> RecipeCapabilities:
        declarations = self.catalog.reference_binder.declarations()
        if (self.base_capabilities.approved_references
                and self.base_capabilities.approved_references != declarations):
            raise SynthesisError("approved reference grants differ from the runtime binder")
        return replace(self.base_capabilities, approved_references=declarations)

    @property
    def capabilities(self) -> RecipeCapabilities:
        return with_admitted_primitives(
            self._reference_capabilities(), self.catalog.capability_declarations(), published_baseline=True,
        )

    def stage(self, request: dict[str, Any], output: Any,
              spec: dict[str, Any], *, allow_identical_replay: bool = False) -> PreparedRecipe:
        role, name = request.get("role"), request.get("name")
        if (not self.base_capabilities.allow_primitive_requests
                or role not in self.base_capabilities.synthesizable_roles):
            raise SynthesisError("requested primitive role is not authorized for synthesis")
        field = ROLE_FIELDS.get(role)
        if field not in self.base_capabilities.allowed_reward_fields:
            raise SynthesisError("synthesis cannot edit a frozen recipe field")
        if output.role != role or output.primitive_name != name:
            raise SynthesisError("Coder identity differs from the requested role/name")
        oracle = request.get("oracle")
        if role == "per_oracle_utility" and oracle not in (spec.get("enabled_oracles") or []):
            raise SynthesisError("utility request must name an enabled optimizer oracle")
        if role != "per_oracle_utility" and oracle is not None:
            raise SynthesisError("only a utility request may bind an oracle")
        generated = PrimitiveRequest(
            role=role, name=name, code=output.code, unit_test=output.unit_test,
            parameters=output.parameters, parameter_schema=output.parameter_schema,
            reference_bindings=tuple(request.get("reference_bindings") or ()), oracle=oracle,
        )
        # Recheck pinned grants before launching any worker, including after a
        # trusted adapter approves another reference since controller creation.
        current_capabilities = self.capabilities
        stage = self.catalog.stage(generated, allow_identical_replay=allow_identical_replay)
        try:
            effective = with_admitted_primitives(
                current_capabilities, {f"{role}/{name}": stage.declaration()},
            )
            updated = copy.deepcopy(spec)
            reference = {"name": name, **copy.deepcopy(output.parameters)}
            if role == "per_oracle_utility":
                if updated.get("oracle_utilities") is None:
                    updated["oracle_utilities"] = {}
                updated["oracle_utilities"][oracle] = reference
            elif role == "cross_oracle_aggregation":
                updated[field] = reference
            else:
                references = updated.get(field) or []
                matches = [index for index, item in enumerate(references)
                           if isinstance(item, dict) and item.get("name") == name]
                if len(matches) > 1:
                    raise SynthesisError("synthesized primitive has duplicate recipe references")
                if matches:
                    references[matches[0]] = reference
                else:
                    references.append(reference)
                updated[field] = references
            return PreparedRecipe(updated, effective, stage)
        except BaseException:
            stage.abort()
            raise

    def screen(self, spec: dict[str, Any], *, parent: dict[str, Any],
               screen: ProposalScreen, source_role_policy: SourceRolePolicy | None,
               available_oracles: list[str], prepared: PreparedRecipe | None = None) -> dict[str, Any]:
        """Execute the actual final recipe, then commit its extension if valid."""
        resolver = prepared.stage.resolve if prepared is not None else self.catalog.resolve
        capabilities = prepared.capabilities if prepared is not None else self.capabilities
        try:
            self._reference_capabilities()
            if prepared is not None:
                request = prepared.stage.request
                field = ROLE_FIELDS[request.role]
                if request.role == "per_oracle_utility":
                    references = [(spec.get(field) or {}).get(request.oracle)]
                elif request.role == "cross_oracle_aggregation":
                    references = [spec.get(field)]
                else:
                    references = spec.get(field) or []
                matches = [item for item in references
                           if isinstance(item, dict) and item.get("name") == request.name]
                if len(matches) != 1:
                    raise SynthesisError(
                        "final recipe must reference the staged primitive exactly once"
                    )
            problems = validate_recipe_capabilities(spec, capabilities, current_spec=parent)
            problems.extend(validate_spec(spec, available_oracles=available_oracles,
                                          source_role_policy=source_role_policy,
                                          primitive_resolver=resolver))
            if problems:
                raise SynthesisError("final recipe failed validation: " + "; ".join(problems))
            actual_screen = replace(screen, primitive_resolver=resolver,
                                    require_training_reference=True,
                                    reject_reference_constants=False)
            problems = actual_screen(spec)
            if problems or actual_screen.last_diagnostics.get("status") != "passed":
                raise SynthesisError("final execution screen failed: " + "; ".join(problems))
            if prepared is not None:
                prepared.stage.publish()
            return {"status": "passed", "diagnostics": copy.deepcopy(actual_screen.last_diagnostics),
                    "extensions": self.catalog.capability_declarations()}
        except BaseException:
            if prepared is not None:
                prepared.abort()
            raise
