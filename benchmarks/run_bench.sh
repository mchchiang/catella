#!/usr/bin/env bash
# Run the full benchmark pipeline: data, tuning, evaluation, plots.
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") OUT [options]

Run the benchmark pipeline (run_bench.py) in OUT. Reruns resume:
make_data is skipped if OUT/datasets.json exists, and evaluate skips
finished units.

Options:
  --quick            Use the small quick-run dataset set
  --nworker N        Number of parallel processes (default 1)
  --env ENV          Conda env for the external tools
                     (default: the run_bench.py default)
  --hiddenfoot PATH  HiddenFoot executable (default: \$HIDDENFOOT)
  --nmol N           Molecules per occupancy plot (default 10)
  -h, --help         Show this message
EOF
}

out=""
quick=()
nworker=1
env=()
hiddenfoot="${HIDDENFOOT:-}"
nmol=10
while [[ $# -gt 0 ]]; do
    case "$1" in
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench=(python "$script_dir/run_bench.py")
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

if [[ -f "$out/datasets.json" ]]; then
    echo "=== make_data skipped: $out/datasets.json exists"
else
    step make_data "${bench[@]}" make_data "$out" ${quick[@]+"${quick[@]}"} \
        ${env[@]+"${env[@]}"} --nworker "$nworker"
fi
step tune "${bench[@]}" tune "$out" ${env[@]+"${env[@]}"} \
    --nworker "$nworker" ${hf[@]+"${hf[@]}"}
step evaluate "${bench[@]}" evaluate "$out" ${env[@]+"${env[@]}"} \
    --nworker "$nworker" ${hf[@]+"${hf[@]}"}
step plot_summary "${bench[@]}" plot_summary "$out"

datasets=$(python -c '
import json, sys
for s in json.load(open(sys.argv[1]))["datasets"]:
    if s["cond"] != "tune":
        print(s["cond"], s["seed"])
' "$out/datasets.json")
pdfs=()
while read -r cond seed; do
    step "plot_occupancy $cond s$seed" "${bench[@]}" plot_occupancy "$out" \
        --cond "$cond" --seed "$seed" --nmol "$nmol"
    pdfs+=("$out/plots/occupancy_${cond}_s${seed}_tuned.pdf")
done <<< "$datasets"

if command -v pdfunite > /dev/null; then
    pdfunite "${pdfs[@]}" "$out/plots/occupancy_all_tuned.pdf"
    echo "=== merged occupancy plots: $out/plots/occupancy_all_tuned.pdf"
else
    echo "=== pdfunite not found; occupancy plots left unmerged"
fi
echo "=== summary: $out/summary.md"
