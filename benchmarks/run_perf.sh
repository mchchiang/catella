#!/usr/bin/env bash
# Run the runtime and memory benchmark: data, measurement, summary, plot.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") OUT [options]

Run the runtime and memory benchmark (run_perf.py) in OUT. Reruns
resume: make_data is skipped if OUT/perf_datasets.json exists, and run
skips finished runs (failed runs are retried).

Options:
  --steps LIST       Comma-separated steps to run, always in pipeline
                     order: make_data, run, summarize, plot; or "all"
                     (default)
  --quick            Use the short quick-run axes (1 repeat)
  --nworker N        Parallel processes for make_data (default 1); runs
                     are always measured one at a time
  --nthread N        Threads (or worker processes) per tool (default 2)
  --nrep N           Repeats per run (default: run_perf.py default)
  --tool LIST        Comma-separated tools to measure (default: all)
  --tuned PATH       tuned_params.json from run_synth.py
                     (default: tool defaults)
  --env ENV          Conda env for the external tools
                     (default: the run_perf.py default)
  --hiddenfoot PATH  HiddenFoot executable (default: \$HIDDENFOOT)
  -h, --help         Show this message

For a single command with extra options, call run_perf.py directly:
  python benchmarks/run_perf.py COMMAND --help
EOF
}

out=""
steps="all"
quick=()
nworker=1
nthread=2
nrep=()
tools=()
tuned=()
env=()
hiddenfoot="${HIDDENFOOT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --quick) quick=(--quick); shift ;;
        --nworker) nworker="$2"; shift 2 ;;
        --nthread) nthread="$2"; shift 2 ;;
        --nrep) nrep=(--nrep "$2"); shift 2 ;;
        --tool)
            for t in ${2//,/ }; do tools+=(--tool "$t"); done
            shift 2 ;;
        --tuned) tuned=(--tuned "$2"); shift 2 ;;
        --env) env=(--env "$2"); shift 2 ;;
        --hiddenfoot) hiddenfoot="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
        *)
            if [[ -n "$out" ]]; then
                echo "unexpected argument: $1" >&2; exit 1
            fi
            out="$1"; shift ;;
    esac
done
if [[ -z "$out" ]]; then
    usage >&2
    exit 1
fi

all_steps="make_data,run,summarize,plot"
selected=","
for s in ${steps//,/ }; do
    case "$s" in
        all) selected+="$all_steps," ;;
        make_data|run|summarize|plot) selected+="$s," ;;
        *) echo "unknown step: $s" >&2; usage >&2; exit 1 ;;
    esac
done

wants() {
    [[ "$selected" == *",$1,"* ]]
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench=(python "$script_dir/run_perf.py")
hf=()
if [[ -n "$hiddenfoot" ]]; then
    hf=(--hiddenfoot "$hiddenfoot")
fi

step() {
    local name="$1"
    shift
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') $name"
    local start=$SECONDS
    "$@"
    echo "=== $name done in $((SECONDS - start)) s"
}

if wants make_data; then
    if [[ -f "$out/perf_datasets.json" ]]; then
        echo "=== make_data skipped: $out/perf_datasets.json exists" \
            "(delete $out to regenerate)"
    else
        step make_data "${bench[@]}" make_data "$out" \
            ${quick[@]+"${quick[@]}"} ${env[@]+"${env[@]}"} \
            --nworker "$nworker"
    fi
fi
if wants run; then
    step run "${bench[@]}" run "$out" ${env[@]+"${env[@]}"} \
        --nthread "$nthread" ${nrep[@]+"${nrep[@]}"} \
        ${tools[@]+"${tools[@]}"} ${tuned[@]+"${tuned[@]}"} \
        ${hf[@]+"${hf[@]}"}
fi
if wants summarize; then
    step summarize "${bench[@]}" summarize "$out"
    echo "=== summary: $out/perf.md"
fi
if wants plot; then
    step plot "${bench[@]}" plot "$out"
    echo "=== plots: $out/plots/perf.pdf"
fi
