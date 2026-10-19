#!/bin/bash
# À lancer SUR LE POD (terminal Jupyter), après un boot complet (ComfyUI UP, custom nodes installés).
# Produit frozen-requirements.txt + wheel SageAttention + commits, puis les pousse dans serverless/ du repo.
set -e

GITHUB_USER="Bigornh2o"
GITHUB_REPO="workflow_bigornh2o"
OUT="/workspace/serverless_export"
REPO_DIR="/tmp/repo_export"

[ -z "$GITHUB_TOKEN" ] && [ -f ~/.bashrc ] && source ~/.bashrc
[ -z "$GITHUB_TOKEN" ] && { echo "❌ GITHUB_TOKEN introuvable (lance d'abord 'save' une fois)"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/wheels"

# 1. Dépendances figées (sans torch/nvidia/triton, déjà dans l'image, ni sageattention, géré en wheel)
/opt/venv/bin/pip freeze --exclude-editable \
  | grep -viE '^(torch|torchvision|torchaudio|triton|nvidia-|sageattention)' \
  | grep -v ' @ ' > "$OUT/frozen-requirements.txt"
echo "✅ frozen-requirements.txt : $(wc -l < "$OUT/frozen-requirements.txt") paquets"

# 2. Wheel SageAttention (même commit que le start script, compilé pour le GPU de ce Pod)
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
[ -d /tmp/SageAttention ] || git clone -q https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
git -C /tmp/SageAttention reset -q --hard 68de379
echo "🔨 Build du wheel SageAttention (~5 min)..."
/opt/venv/bin/pip wheel /tmp/SageAttention --no-deps --no-build-isolation -w "$OUT/wheels" -q
ls -lh "$OUT/wheels"
for w in "$OUT"/wheels/*.whl; do
  [ "$(stat -c%s "$w")" -gt 95000000 ] && { echo "❌ $w dépasse 95 Mo (limite GitHub 100 Mo)"; exit 1; }
done

# 3. Versions figées : ComfyUI + custom nodes (traçabilité, rien n'est mis à jour en Serverless)
git -C /workspace/ComfyUI rev-parse HEAD > "$OUT/comfyui_commit.txt"
for d in /workspace/ComfyUI/custom_nodes/*/; do
  [ -d "$d/.git" ] && echo "$(basename "$d") $(git -C "$d" rev-parse --short HEAD) $(git -C "$d" remote get-url origin | sed 's#//[^@/]*@#//#')"
done > "$OUT/custom_nodes_commits.txt"

# 4. Push dans serverless/
rm -rf "$REPO_DIR"
git clone -q "https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$GITHUB_REPO.git" "$REPO_DIR"
cd "$REPO_DIR"
git config user.email "$GITHUB_USER@github.com"; git config user.name "$GITHUB_USER"
mkdir -p serverless/wheels
cp "$OUT/frozen-requirements.txt" "$OUT/comfyui_commit.txt" "$OUT/custom_nodes_commits.txt" serverless/
cp "$OUT"/wheels/*.whl serverless/wheels/
git add serverless/
if git diff --cached --quiet; then echo "ℹ️ Rien à pousser."; else
  git commit -q -m "serverless: deps figées + wheel SageAttention $(date '+%Y-%m-%d %H:%M')"
  git push -q origin main && echo "✅ Poussé dans serverless/"
fi
