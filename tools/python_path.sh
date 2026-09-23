# Sourced, not run: sets PY to the buzzdetect-train python for this machine.
# paths.local.json's "python" key wins; otherwise the env under conda's base
# (or ~/anaconda3 when conda isn't on PATH, as in a bare non-login shell).
_root=$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.." && pwd)
PY=$(sed -nE 's/^[[:space:]]*"python"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/p' \
       "$_root/paths.local.json" 2>/dev/null | head -n 1)
if [ -z "$PY" ]; then
  for _base in "$(conda info --base 2>/dev/null)" "$HOME/anaconda3"; do
    [ -n "$_base" ] && [ -x "$_base/envs/buzzdetect-train/bin/python" ] \
      && { PY="$_base/envs/buzzdetect-train/bin/python"; break; }
  done
fi
[ -x "${PY:-}" ] || { echo "no buzzdetect-train python found; set \"python\" in paths.local.json" >&2; exit 1; }
unset _root _base
