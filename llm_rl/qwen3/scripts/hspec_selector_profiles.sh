#!/usr/bin/env bash

# S18 selector release profiles.  Call this after PROJECT_ROOT is defined and
# before exporting/forwarding individual HSPEC selector variables.
hspec_apply_selector_profile() {
    local profile="${HSPEC_SELECTOR_PROFILE:-production_p3a}"
    local root="${PROJECT_ROOT:?PROJECT_ROOT must be set before selector profile}"

    export HSPEC_SELECTOR_PROFILE="${profile}"
    case "${profile}" in
        production_p3a)
            # S17 memory-policy-v3 global default: the exact P3A envelope that
            # passed S13 execution authorization and the two-task 30B gate.
            export HSPEC_SELECT_MODE=topk_utility
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
            export HSPEC_SELECT_MODEL_PATH="${root}/HSpec_research_doc/HSpec_draft_delect_optim/s12_to_s13_transition/candidate/transition_candidate.json"
            export HSPEC_SELECT_MODEL_SHA256=c3982dd40b1124d14c942dee1a010c7c441bbbbdff30d5f86b1bc1c025d0f869
            export HSPEC_SELECT_MODEL_VERSION=s12-transition-fixed-theta-bias-v1
            export HSPEC_SELECT_PROMOTION_GATE_PATH="${root}/HSpec_research_doc/HSpec_draft_delect_optim/s13_patch3a_utility/artifacts/s13_entry_gate.json"
            export HSPEC_SELECT_PROMOTION_GATE_SHA256=5afb4a4c4c57d9032775832b69a88b6ae571d9dce90edf4aaab889e5453cc6b0
            export HSPEC_SELECT_EXECUTION_GATE_PATH="${root}/outputs/hspec_draft_select_research/s13_patch3a_utility/manual_v2_20260811T065928Z/target_shadow_analysis/gate_result.json"
            export HSPEC_SELECT_EXECUTION_GATE_SHA256=24e694d7a3d4cf28929c72d00d79d3e039365b86272e372b56541a8e4255e6e1
            export HSPEC_SELECT_EXECUTION_LEVEL=performance
            export HSPEC_SELECT_ALLOW_EXECUTE=1
            export HSPEC_SELECT_D2H_STRATEGY=pinned_two_async
            export HSPEC_SELECT_R1_COMPARE_EVERY_BATCHES=0
            export HSPEC_MAX_DRAFT_TOKENS_PER_BATCH=384
            export HSPEC_S13_FASTPATH_VERSION=p3-utility-first-batch-v1
            export HSPEC_S14_MODE=off
            ;;
        production_r1|r1_rollback)
            # Model-free first rollback. Keep production_r1 as a compatibility
            # alias for existing operator automation.
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
            export HSPEC_SELECT_R1_COMPARE_EVERY_BATCHES=0
            export HSPEC_MAX_DRAFT_TOKENS_PER_BATCH=384
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
            export HSPEC_SELECT_R1_COMPARE_EVERY_BATCHES=0
            export HSPEC_MAX_DRAFT_TOKENS_PER_BATCH=0
            export HSPEC_S13_FASTPATH_VERSION=
            export HSPEC_S14_MODE=off
            ;;
        p3a_shadow)
            # Diagnostic mirror of P3A. It computes but never executes the new
            # decision; missing/tampered artifacts fail closed.
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
