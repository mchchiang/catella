#!/usr/bin/env bash
# Run the full benchmark pipeline: data, tuning, evaluation, plots.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") OUT [options]

Run the benchmark pipeline (run_synth.py) in OUT. Reruns resume:
make_data is skipped if OUT/datasets.json exists, and evaluate skips
finished units.

Options:
  --steps LIST       Comma-separated steps to run, always in pipeline
                     order: make_data, tune, evaluate, rescore,
                     plot_summary, plot_reliability, plot_occupancy; or
                     "all" (default; every step but rescore) or "plot"
                     (the three plot steps). rescore scores the saved
                     calls again without running tools
  --quick            Use the small quick-run dataset set
  --nworker N        Number of parallel processes (default 1)
  --env ENV          Conda env for the external tools
                     (default: the run_synth.py default)
  --hiddenfoot PATH  HiddenFoot executable (default: \$HIDDENFOOT)
  --nmol N           Molecules per occupancy plot (default 10)
  -h, --help         Show this message

For a single command with extra options, call run_synth.py directly:
  python benchmarks/run_synth.py COMMAND --help
EOF
}

out=""
steps="all"
quick=()
nworker=1
env=()
hiddenfoot="${HIDDENFOOT:-}"
nmol=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --quick) quick=(--quick); shift ;;
        --nworker) nworker="$2"; shift 2 ;;
        --env) env=(--env "$2"); shift 2 ;;
        --hiddenfoot) hiddenfoot="$2"; shift 2 ;;
        --nmol) nmol="$2"; shift 2 ;;
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

plot_steps="plot_summary,plot_reliability,plot_occupancy"
all_steps="make_data,tune,evaluate,$plot_steps"
selected=","
for s in ${steps//,/ }; do
    case "$s" in
        all) selected+="$all_steps," ;;
        plot) selected+="$plot_steps," ;;
        make_data|tune|evaluate|rescore|plot_summary|plot_reliability)
            selected+="$s," ;;
        plot_occupancy) selected+="$s," ;;
        *) echo "unknown step: $s" >&2; usage >&2; exit 1 ;;
    esac
done

wants() {
    [[ "$selected" == *",$1,"* ]]
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench=(python "$script_dir/run_synth.py")
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
    if [[ -f "$out/datasets.json" ]]; then
        echo "=== make_data skipped: $out/datasets.json exists" \
            "(delete $out to regenerate)"
    else
        step make_data "${bench[@]}" make_data "$out" \
            ${quick[@]+"${quick[@]}"} ${env[@]+"${env[@]}"} \
            --nworker "$nworker"
    fi
fi
if wants tune; then
    step tune "${bench[@]}" tune "$out" ${env[@]+"${env[@]}"} \
        --nworker "$nworker" ${hf[@]+"${hf[@]}"}
fi
if wants evaluate; then
    step evaluate "${bench[@]}" evaluate "$out" ${env[@]+"${env[@]}"} \
        --nworker "$nworker" ${hf[@]+"${hf[@]}"}
fi
if wants rescore; then
    step rescore "${bench[@]}" rescore "$out"
fi
if wants plot_summary; then
    step plot_summary "${bench[@]}" plot_summary "$out"
fi
if wants plot_reliability; then
    step plot_reliability "${bench[@]}" plot_reliability "$out"
fi

if wants plot_occupancy; then
    datasets=$(python -c '
import json, sys
for s in json.load(open(sys.argv[1]))["datasets"]:
    if s["cond"] != "tune":
        print(s["cond"], s["seed"])
' "$out/datasets.json")
    pdfs=()
    while read -r cond seed; do
        step "plot_occupancy $cond s$seed" "${bench[@]}" plot_occupancy \
            "$out" --cond "$cond" --seed "$seed" --nmol "$nmol"
        pdfs+=("$out/plots/occupancy_${cond}_s${seed}_tuned.pdf")
    done <<< "$datasets"
    if command -v pdfunite > /dev/null; then
        pdfunite "${pdfs[@]}" "$out/plots/occupancy_all_tuned.pdf"
        echo "=== merged occupancy plots:" \
            "$out/plots/occupancy_all_tuned.pdf"
    else
        echo "=== pdfunite not found; occupancy plots left unmerged"
    fi
fi
if wants evaluate; then
    echo "=== summary: $out/summary.md"
fi
