# Image Serverless : dérive de l'image Pod existante.
# On y garde : /opt/venv, /constraints.txt, /flux_resolution_cal_node.py (nœud patché).
# Le volume réseau (ComfyUI, custom nodes, modèles, LoRA) n'est PAS dans l'image.
FROM bigornh2o/comfyui-custom:krea2

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# ── rclone (upload Drive) ─────────────────────────────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl unzip ca-certificates \
 && curl -fsSL https://rclone.org/install.sh | bash \
 && rm -rf /var/lib/apt/lists/*

# ── SDK Runpod ────────────────────────────────────────────────────────────────
RUN /opt/venv/bin/pip install -c /constraints.txt runpod

# ── Dépendances figées : remplace le pip install du start script ──────────────
# Généré par export_from_pod.sh (pip freeze du venv du Pod, après un boot complet).
COPY frozen-requirements.txt /tmp/frozen-requirements.txt
RUN /opt/venv/bin/pip install -c /constraints.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -r /tmp/frozen-requirements.txt

# ── SageAttention : wheel compilé sur le Pod (commit 68de379), plus de build au boot
COPY wheels/ /tmp/wheels/
RUN /opt/venv/bin/pip install --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

COPY handler.py /handler.py

# Pas de Jupyter, pas de start_bigornh2o.sh, pas de git pull ni de pip au boot
ENTRYPOINT []
CMD ["/opt/venv/bin/python", "-u", "/handler.py"]
