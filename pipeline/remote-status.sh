#!/usr/bin/env bash
ssh "${AYEKOO_REMOTE_HOST:-h200}" "bash ${AYEKOO_REMOTE_DIR:-/mnt/volume_d2wey28/projects/ayekoo-videos}/status.sh"
