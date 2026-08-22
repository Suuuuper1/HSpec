#!/usr/bin/env bash

# S18 selector release profiles.  Call this after PROJECT_ROOT is defined and
# before exporting/forwarding individual HSPEC selector variables.
hspec_apply_selector_profile() {
    local profile="${HSPEC_SELECTOR_PROFILE:-production_r1}"
    local root="${PROJECT_ROOT:?PROJECT_ROOT must be set before selector profile}"

    export HSPEC_SELECTOR_PROFILE="${profile}"
    case "${profile}" in
        production_r1)
            # S17 global default: frozen S6/S10 R1, with all later research
            # components disabled and no model or gate dependency.
            export HSPEC_SELECT_MODE=topk_position
            export HSPEC_SELECT_TOPK=8
            export HSPEC_SELECT_SIM_MODE=cosine
            export HSPEC_SELECT_RELATIVE_RADIUS=0.0001
            export HSPEC_SELECT_SUFFIX_CAP=8
            export HSPEC_SELECT_POSITION_MODE=none
            export HSPEC_SELECT_SHADOW=0
            export HSPEC_SELECT_RELATIVE_WEIGHT=7.719409849724556
            export HSPEC_SELECT_SUFFIX_WEIGHT=0.6382322349890022
            export HSPEC_SELECT_POSITION_EXACT_WEIGHT=0
            export HSPEC_SELECT_POSITION_LOG_WEIGHT=0
            export HSPEC_SELECT_UTILITY_THRESHOLD=-1e30
            export HSPEC_SELECT_MODEL_PATH=
            export HSPEC_SELECT_MODEL_SHA256=
            export HSPEC_SELECT_MODEL_VERSION=
            export HSPEC_SELECT_PROMOTION_GATE_PATH=
            export HSPEC_SELECT_PROMOTION_GATE_SHA256=
            export HSPEC_SELECT_EXECUTION_GATE_PATH=
            export HSPEC_SELECT_EXECUTION_GATE_SHA256=
            export HSPEC_SELECT_EXECUTION_LEVEL=
            export HSPEC_SELECT_ALLOW_EXECUTE=0
            export HSPEC_S13_FASTPATH_VERSION=
            export HSPEC_S14_MODE=off
            ;;
        hardmax_rollback)
            # P0 is independent of learned artifacts and remains the emergency
            # rollback even when a selector/model/schema validation fails.
            export HSPEC_SELECT_MODE=hardmax
            export HSPEC_SELECT_TOPK=1
            export HSPEC_SELECT_SIM_MODE=raw
            export HSPEC_SELECT_POSITION_MODE=none
            export HSPEC_SELECT_SHADOW=0
            export HSPEC_SELECT_MODEL_PATH=
            export HSPEC_SELECT_MODEL_SHA256=
            export HSPEC_SELECT_MODEL_VERSION=
            export HSPEC_SELECT_PROMOTION_GATE_PATH=
            export HSPEC_SELECT_PROMOTION_GATE_SHA256=
            export HSPEC_SELECT_EXECUTION_GATE_PATH=
            export HSPEC_SELECT_EXECUTION_GATE_SHA256=
            export HSPEC_SELECT_EXECUTION_LEVEL=
            export HSPEC_SELECT_ALLOW_EXECUTE=0
            export HSPEC_S13_FASTPATH_VERSION=
            export HSPEC_S14_MODE=off
            ;;
        p3a_shadow)
            # Research-only diagnostic profile.  It computes the frozen P3A
            # decision but never executes it; missing/tampered artifacts cause
            # the existing init-time R1/hardmax fallback.
            export HSPEC_SELECT_MODE=topk_utility
            export HSPEC_SELECT_TOPK=8
            export HSPEC_SELECT_SIM_MODE=cosine
            export HSPEC_SELECT_RELATIVE_RADIUS=0.0001
            export HSPEC_SELECT_SUFFIX_CAP=8
            export HSPEC_SELECT_POSITION_MODE=none
            export HSPEC_SELECT_SHADOW=1
            export HSPEC_SELECT_ALLOW_EXECUTE=0
            export HSPEC_SELECT_MODEL_PATH="${root}/HSpec_research_doc/HSpec_draft_delect_optim/s12_to_s13_transition/candidate/transition_candidate.json"
            export HSPEC_SELECT_MODEL_SHA256=c3982dd40b1124d14c942dee1a010c7c441bbbbdff30d5f86b1bc1c025d0f869
            export HSPEC_SELECT_MODEL_VERSION=s12-transition-fixed-theta-bias-v1
            export HSPEC_SELECT_PROMOTION_GATE_PATH="${root}/HSpec_research_doc/HSpec_draft_delect_optim/s13_patch3a_utility/artifacts/s13_entry_gate.json"
            export HSPEC_SELECT_PROMOTION_GATE_SHA256=5afb4a4c4c57d9032775832b69a88b6ae571d9dce90edf4aaab889e5453cc6b0
            export HSPEC_SELECT_EXECUTION_GATE_PATH=
            export HSPEC_SELECT_EXECUTION_GATE_SHA256=
            export HSPEC_SELECT_EXECUTION_LEVEL=
            export HSPEC_S13_FASTPATH_VERSION=p3-utility-first-batch-v1
            export HSPEC_S14_MODE=off
            ;;
        custom)
            # Registered research runners use this profile and must set their
            # arm explicitly.  Defaults remain fail-closed for ad-hoc custom
            # launches with incomplete settings.
            export HSPEC_SELECT_MODE="${HSPEC_SELECT_MODE:-hardmax}"
            export HSPEC_SELECT_TOPK="${HSPEC_SELECT_TOPK:-1}"
            export HSPEC_SELECT_SIM_MODE="${HSPEC_SELECT_SIM_MODE:-raw}"
            ;;
        *)
            echo "ERROR: unknown HSPEC_SELECTOR_PROFILE=${profile}" >&2
            return 2
            ;;
    esac
}
