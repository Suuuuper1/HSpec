#!/usr/bin/env bash

# Shared HSpec store lifecycle helpers for training entry scripts.
# The helpers run only at process startup, before Ray workers/actors are
# created. They intentionally stay out of the decode/build hot paths.

hspec_configure_store_lifecycle() {
    local default_mode="${1:-clean}"
    export HSPEC_RUN_NAME="${HSPEC_RUN_NAME:-${RUN_NAME:-hspec_run}}"
    export HSPEC_RUN_UID="${HSPEC_RUN_UID:-$(date -u '+%Y%m%dT%H%M%SZ')_$$}"
    export HSPEC_STORE_ISOLATION_MODE="${HSPEC_STORE_ISOLATION_MODE:-${default_mode}}"

    case "${HSPEC_STORE_ISOLATION_MODE}" in
        clean|unique|reuse)
            ;;
        *)
            echo "ERROR: invalid HSPEC_STORE_ISOLATION_MODE=${HSPEC_STORE_ISOLATION_MODE}; expected clean, unique, or reuse." >&2
            return 2
            ;;
    esac

    if [ "${HSPEC_STORE_ISOLATION_MODE}" = "unique" ]; then
        export HSPEC_STORE_DIR="${HSPEC_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_store/${HSPEC_RUN_NAME}/${HSPEC_RUN_UID}}"
        export HSPEC_TABLE_STORE_DIR="${HSPEC_TABLE_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_table_store/${HSPEC_RUN_NAME}/${HSPEC_RUN_UID}}"
        export HSPEC_BUILD_ACTOR_NAME_PREFIX="${HSPEC_BUILD_ACTOR_NAME_PREFIX:-hspec_build_${HSPEC_RUN_NAME}_${HSPEC_RUN_UID}}"
    else
        export HSPEC_STORE_DIR="${HSPEC_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_store/${HSPEC_RUN_NAME}}"
        export HSPEC_TABLE_STORE_DIR="${HSPEC_TABLE_STORE_DIR:-${PROJECT_ROOT}/outputs/hspec_table_store/${HSPEC_RUN_NAME}}"
        export HSPEC_BUILD_ACTOR_NAME_PREFIX="${HSPEC_BUILD_ACTOR_NAME_PREFIX:-hspec_build_${HSPEC_RUN_NAME}}"
    fi

    local default_clean=0
    if [ "${HSPEC_STORE_ISOLATION_MODE}" = "clean" ]; then
        default_clean=1
    fi
    export HSPEC_CLEAN_STORE_ON_START="${HSPEC_CLEAN_STORE_ON_START:-${default_clean}}"
    export HSPEC_CLEAN_RAW_STORE_ON_START="${HSPEC_CLEAN_RAW_STORE_ON_START:-${HSPEC_CLEAN_STORE_ON_START}}"
    export HSPEC_CLEAN_TABLE_STORE_ON_START="${HSPEC_CLEAN_TABLE_STORE_ON_START:-${HSPEC_CLEAN_STORE_ON_START}}"
    export HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT="${HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT:-0}"
    export HSPEC_REQUIRE_FRESH_TABLE_STORE="${HSPEC_REQUIRE_FRESH_TABLE_STORE:-${HSPEC_CLEAN_TABLE_STORE_ON_START}}"
}

_hspec_python_bin() {
    if [ -n "${PYTHON:-}" ]; then
        printf '%s\n' "${PYTHON}"
    elif command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
    else
        printf '%s\n' "python"
    fi
}

_hspec_real_path() {
    "$(_hspec_python_bin)" -c 'import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' "$1"
}

hspec_safe_clean_dir() {
    local target="$1"
    local label="$2"
    if [ -z "${target}" ] || [ "${target}" = "/" ]; then
        echo "Refusing to clean empty/root HSpec ${label} dir: ${target}" >&2
        return 1
    fi

    local target_abs
    local raw_root_abs
    local table_root_abs
    target_abs="$(_hspec_real_path "${target}")"
    raw_root_abs="$(_hspec_real_path "${PROJECT_ROOT}/outputs/hspec_store")"
    table_root_abs="$(_hspec_real_path "${PROJECT_ROOT}/outputs/hspec_table_store")"

    if [ -z "${target_abs}" ] || [ "${target_abs}" = "/" ]; then
        echo "Refusing to clean empty/root HSpec ${label} dir: ${target_abs}" >&2
        return 1
    fi
    if [ "${target_abs}" = "${raw_root_abs}" ] || [ "${target_abs}" = "${table_root_abs}" ]; then
        echo "Refusing to clean top-level HSpec ${label} root: ${target_abs}" >&2
        return 1
    fi

    case "${target_abs}/" in
        "${raw_root_abs}/"*|"${table_root_abs}/"*)
            echo "Cleaning HSpec ${label} dir: ${target_abs}"
            rm -rf -- "${target_abs}"
            mkdir -p -- "${target_abs}"
            ;;
        *)
            if [ "${HSPEC_ALLOW_CLEAN_OUTSIDE_PROJECT}" = "1" ]; then
                echo "Cleaning HSpec ${label} dir outside project outputs: ${target_abs}"
                rm -rf -- "${target_abs}"
                mkdir -p -- "${target_abs}"
            else
                echo "Refusing to clean HSpec ${label} dir outside project outputs: ${target_abs}" >&2
                return 1
            fi
            ;;
    esac
}

hspec_maybe_clean_store_dirs() {
    if [ "${USE_HSPEC_DECODE:-1}" = "0" ]; then
        return 0
    fi

    if [ "${HSPEC_CLEAN_RAW_STORE_ON_START}" != "0" ]; then
        hspec_safe_clean_dir "${HSPEC_STORE_DIR}" "raw store"
    else
        mkdir -p -- "${HSPEC_STORE_DIR}"
    fi

    if [ "${HSPEC_CLEAN_TABLE_STORE_ON_START}" != "0" ]; then
        hspec_safe_clean_dir "${HSPEC_TABLE_STORE_DIR}" "table store"
    else
        mkdir -p -- "${HSPEC_TABLE_STORE_DIR}"
    fi
}
