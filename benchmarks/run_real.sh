#!/usr/bin/env bash
# Run the real-data benchmark pipeline: prepare, evaluate, score, plot.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") OUT [options]

Run the real-data benchmark pipeline (run_real.py) in OUT. Reruns resume:
prepare is skipped if OUT/annotation.json exists, and evaluate skips
finished units.

Options:
  --steps LIST       Comma-separated steps to run, always in pipeline
                     order: prepare, evaluate, score, plot; or "all"
                     (default)
  --raw-dir DIR      Directory with chromatin.tsv.gz, methylated.tsv.gz,
                     unmethylated.tsv.gz and segments.fa (needed by
                     prepare)
  --tuned FILE       tuned_params.json of a synthetic benchmark run
                     (needed by evaluate)
  --nmol N           Molecules per sample
                     (default: the run_real.py default)
  --ntrack N         Molecules shown as tracks
                     (default: the run_real.py default)
  --nworker N        Number of parallel processes (default 1)
  --env ENV          Conda env for the external tools
                     (default: the run_real.py default)
  --hiddenfoot PATH  HiddenFoot executable (default: \$HIDDENFOOT)
  -h, --help         Show this message

For a single command with extra options, call run_real.py directly:
  python benchmarks/run_real.py COMMAND --help
EOF
}

out=""
steps="all"
raw_dir=""
tuned=""
nmol=()
ntrack=()
nworker=1
env=()
hiddenfoot="${HIDDENFOOT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --raw-dir) raw_dir="$2"; shift 2 ;;
        --tuned) tuned="$2"; shift 2 ;;
        --nmol) nmol=(--nmol "$2"); shift 2 ;;
        --ntrack) ntrack=(--nmol "$2"); shift 2 ;;
        --nworker) nworker="$2"; shift 2 ;;
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

selected=","
for s in ${steps//,/ }; do
    case "$s" in
        all) selected+="prepare,evaluate,score,plot," ;;
        prepare|evaluate|score|plot) selected+="$s," ;;
        *) echo "unknown step: $s" >&2; usage >&2; exit 1 ;;
    esac
done

wants() {
    [[ "$selected" == *",$1,"* ]]
}

run_prepare=0
if wants prepare && [[ ! -f "$out/annotation.json" ]]; then
    run_prepare=1
fi
if [[ "$run_prepare" == 1 && -z "$raw_dir" ]]; then
    echo "prepare needs --raw-dir" >&2
    exit 1
fi
if wants evaluate && [[ -z "$tuned" ]]; then
    echo "evaluate needs --tuned" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
real=(python "$script_dir/run_real.py")
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

if wants prepare; then
    if [[ "$run_prepare" == 1 ]]; then
        step prepare "${real[@]}" prepare "$raw_dir" "$out" \
            ${nmol[@]+"${nmol[@]}"} ${env[@]+"${env[@]}"}
    else
        echo "=== prepare skipped: $out/annotation.json exists" \
            "(delete $out to regenerate)"
    fi
fi
if wants evaluate; then
    step evaluate "${real[@]}" evaluate "$out" --tuned "$tuned" \
        ${env[@]+"${env[@]}"} --nworker "$nworker" ${hf[@]+"${hf[@]}"}
fi
if wants score; then
    step score "${real[@]}" score "$out"
    echo "=== summary: $out/real_summary.md"
fi
if wants plot; then
    step plot "${real[@]}" plot "$out" ${ntrack[@]+"${ntrack[@]}"}
fi
