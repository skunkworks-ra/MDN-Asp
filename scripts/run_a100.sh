#!/usr/bin/env bash
# End-to-end A100 execution script for mdn-asp publication runs.
#
# Subcommands:
#   preflight    Run unit test suite, verify GPU + environment, check dataset/checkpoint paths
#   tier1_eval   Run Tier-1 loop fixes (NNLS refit, UV-coupling guard, q-space DDS, AIC DOF) on existing checkpoints
#   retrain_all  Retrain MDN-Asp v2 (von Mises PA + uniform point PA + correlated noise), Diffusion P v2, Diffusion LD v2 (conv stem + partial-clean residuals)
#   tier2_eval   Evaluate v2 checkpoints on sim_recovery, eval_diffusion, sim_jet, 3C391, G55, Cygnus A, and export HF folders
#   eval_jet     Compute support-masked on-source / core / off-source flux table for sim_jet runs
#   all          preflight -> tier1_eval -> retrain_all -> tier2_eval
#
# Override paths via environment variables before running:
#   REPO_DIR, PIXI_ENV, DEVICE, OUT_ROOT,
#   PSF_ROOT, CORPUS_PSF_DIR, AGN_NPZ,
#   MS_3C391, MASK_3C391, MS_G55, MS_CYGA, SIM_JET_DIR,
#   CKPT_MDN_V1, CKPT_MDN_256_V1, CKPT_B2_V1, CKPT_A3_V1

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PIXI_ENV="${PIXI_ENV:-gpu}"
DEVICE="${DEVICE:-cuda:0}"

# Default data/checkpoint locations (override on the A100 host as needed)
DATA_FAST="${DATA_FAST:-/mnt/fast/mad_clean_runs}"
PSF_ROOT="${PSF_ROOT:-${MDN_ASP_PSF_ROOT:-$DATA_FAST/mdn_asp_v2_relative/psfroot}}"
CORPUS_PSF_DIR="${CORPUS_PSF_DIR:-${MDN_ASP_CORPUS_PSF_DIR:-/mnt/Data/Data/corpus_fits}}"
AGN_NPZ="${AGN_NPZ:-${MDN_ASP_AGN_NPZ:-$DATA_FAST/mdn_asp_split/trecs_counts/trecs_counts.npz}}"

MS_DIR="${MS_DIR:-/home/pjaganna/Data/measurement_sets}"
MS_3C391="${MS_3C391:-$MS_DIR/3c391_ctm_mosaic_10s_spw0.ms}"
MASK_3C391="${MASK_3C391:-$DATA_FAST/mdn_asp_split/replicate/3c391_asp_mask_grow2beam.mask}"
MS_G55="${MS_G55:-$MS_DIR/SNR_G55_10s.calib.ms}"
MS_CYGA="${MS_CYGA:-$MS_DIR/CYG-2052-2MHz.ms}"
SIM_JET_DIR="${SIM_JET_DIR:-$DATA_FAST/mdn_asp_split/sim_jet}"

# Existing v1 checkpoints (accepts local .pt files or downloaded HF folders)
HF_DIR="${HF_DIR:-$REPO_DIR/models/mdn-asp}"
CKPT_MDN_V1="${CKPT_MDN_V1:-$DATA_FAST/mdn_asp_split/train_px0.5_rotfix/best.pt}"
CKPT_MDN_256_V1="${CKPT_MDN_256_V1:-$DATA_FAST/mdn_asp_split/train/best.pt}"
CKPT_B2_V1="${CKPT_B2_V1:-$DATA_FAST/mdn_asp_split/diff_prior/B2_prior/ckpt_0040000.pt}"
CKPT_A3_V1="${CKPT_A3_V1:-$DATA_FAST/mdn_asp_split/diff_dict/A3_full_lam0/best.pt}"
if [[ ! -e "$CKPT_MDN_V1" && -d "$HF_DIR" ]]; then CKPT_MDN_V1="$HF_DIR"; fi
if [[ ! -e "$CKPT_MDN_256_V1" && -d "$HF_DIR/mdn_asp_train256" ]]; then CKPT_MDN_256_V1="$HF_DIR/mdn_asp_train256"; fi
if [[ ! -e "$CKPT_B2_V1" && -d "$HF_DIR/diffusion_B2" ]]; then CKPT_B2_V1="$HF_DIR/diffusion_B2"; fi
if [[ ! -e "$CKPT_A3_V1" && -d "$HF_DIR/diffusion_A3" ]]; then CKPT_A3_V1="$HF_DIR/diffusion_A3"; fi

OUT_ROOT="${OUT_ROOT:-$DATA_FAST/mdn_asp_split/paper_v2_a100}"
STATUS_FILE="$OUT_ROOT/status.txt"

PY_CMD=(pixi run -e "$PIXI_ENV" python -u)

log_status() {
  local msg="$1"
  mkdir -p "$OUT_ROOT"
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $msg" | tee -a "$STATUS_FILE"
}

run_job() {
  local name="$1"
  local dir="$2"
  shift 2
  mkdir -p "$dir"
  (
    cd "$REPO_DIR"
    git rev-parse HEAD > "$dir/git.txt"
    git status --short >> "$dir/git.txt"
  )
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd \"$REPO_DIR\""
    printf '%q ' "${PY_CMD[@]}" "$@"
    echo
  } > "$dir/cmd.sh"
  chmod +x "$dir/cmd.sh"

  log_status "START $name -> $dir"
  set +e
  (cd "$dir" && bash "$dir/cmd.sh" > "$dir/run.log" 2>&1)
  local rc=$?
  set -e
  log_status "DONE  $name exit=$rc"
  if [[ $rc -ne 0 ]]; then
    echo "ERROR: $name failed (exit $rc). Tail of $dir/run.log:" >&2
    tail -n 40 "$dir/run.log" >&2 || true
    return $rc
  fi
}

preflight() {
  log_status "PREFLIGHT starting in $REPO_DIR on $DEVICE"
  (cd "$REPO_DIR" && pixi run test)
  (
    cd "$REPO_DIR"
    "${PY_CMD[@]}" -c "
import torch
dev = torch.device('$DEVICE')
print('[preflight] torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())
if dev.type == 'cuda':
    print('[preflight] device:', torch.cuda.get_device_name(dev),
          f'{torch.cuda.get_device_properties(dev).total_memory / 2**30:.1f} GiB')
"
  )
  for label_path in \
    "PSF_ROOT:$PSF_ROOT" \
    "CORPUS_PSF_DIR:$CORPUS_PSF_DIR" \
    "AGN_NPZ:$AGN_NPZ" \
    "MS_3C391:$MS_3C391" \
    "MASK_3C391:$MASK_3C391" \
    "MS_G55:$MS_G55" \
    "MS_CYGA:$MS_CYGA" \
    "SIM_JET_DIR:$SIM_JET_DIR" \
    "CKPT_MDN_V1:$CKPT_MDN_V1" \
    "CKPT_B2_V1:$CKPT_B2_V1" \
    "CKPT_A3_V1:$CKPT_A3_V1"
  do
    local lbl="${label_path%%:*}"
    local pth="${label_path#*:}"
    if [[ -e "$pth" ]]; then
      echo "[preflight] OK   $lbl = $pth"
    else
      echo "[preflight] MISS $lbl = $pth"
    fi
  done
  log_status "PREFLIGHT passed"
}

run_imaging_suite() {
  local tag="$1"
  local mdn_128="$2"
  local mdn_256="$3"
  local diff_p="$4"
  local diff_ld="$5"
  local suite_dir="$OUT_ROOT/$tag"
  mkdir -p "$suite_dir"

  # 1. sim_jet (512 px, 12 arcsec, no mask, threshold 0.05 Jy/beam)
  if [[ -d "$SIM_JET_DIR/sim_data_VLA.ms" ]]; then
    run_job "${tag}_sim_jet_mdn" "$suite_dir/sim_jet_mdn" \
      scripts/run_imaging.py \
      --vis "$SIM_JET_DIR/sim_data_VLA.ms" --imagename jet \
      --out_dir "$suite_dir/sim_jet_mdn" \
      --imsize 512 --cell 12arcsec --gridder standard --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask_radius_px 0 --n_sigma 3 --no_psf_floor --global_sigma --global_threshold 0.05 \
      --loop_gain 0.1 --max_major 10 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --ckpt "$mdn_128" --sigma_min_px 0.5 --sigma_max_px 11.2 --flux_clip \
      --mdn_density_mode --mdn_profile_amp --mdn_delta_fallback --uv_coupling_min 0.25 \
      --refit_exact --major_before_refit --refit_all --refit_method nnls \
      --refit_gain 0.5 --refit_nullspace_lam 1e-4 --error_map --n_posterior_samples 200 \
      --device "$DEVICE"

    run_job "${tag}_sim_jet_B2" "$suite_dir/sim_jet_B2" \
      scripts/run_imaging.py \
      --vis "$SIM_JET_DIR/sim_data_VLA.ms" --imagename jet \
      --out_dir "$suite_dir/sim_jet_B2" \
      --imsize 512 --cell 12arcsec --gridder standard --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask_radius_px 0 --n_sigma 3 --no_psf_floor --global_sigma --global_threshold 0.05 \
      --loop_gain 0.2 --max_major 10 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --diffusion_ckpt "$diff_p" --diff_mp --diff_n_draws 10 --diff_seed 0 \
      --fit_unmasked --diff_unmask_output --diff_taper --diff_q_space --diff_dof_pen 2.0 --diff_briggs_psd \
      --refit_exact --major_before_refit --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-4 \
      --diff_rebuild_drop 0.2 --diff_amp_error --diff_save_atoms \
      --device "$DEVICE"

    eval_jet "$suite_dir"
  fi

  # 2. 3C391 (480 px, 2.5 arcsec, mosaic, masked + unmasked comparison)
  if [[ -d "$MS_3C391" ]]; then
    run_job "${tag}_3c391_mdn" "$suite_dir/3c391_mdn" \
      scripts/run_imaging.py \
      --vis "$MS_3C391" --imagename 3c391 --field '3C391*' \
      --out_dir "$suite_dir/3c391_mdn" \
      --imsize 480 --cell 2.5arcsec --gridder mosaic --wprojplanes -1 --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask "$MASK_3C391" --mask_radius_px 200 --n_sigma 3.0 --no_psf_floor --global_sigma --global_threshold 0.001 \
      --loop_gain 0.1 --max_major 10 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --ckpt "$mdn_128" --sigma_min_px 0.5 --sigma_max_px 96.0 --fit_unmasked \
      --mdn_density_mode --mdn_profile_amp --mdn_delta_fallback --uv_coupling_min 0.15 \
      --refit_exact --major_before_refit --refit_all --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
      --error_map --device "$DEVICE"

    for mode_pair in "B2|$diff_p||10" "A3|$diff_p|$diff_ld|0" "B2_A3|$diff_p|$diff_ld|10"; do
      IFS='|' read -r mname p_ck ld_ck n_b <<< "$mode_pair"
      n_b="${n_b:-10}"
      extra_ld=()
      if [[ -n "$ld_ck" ]]; then
        extra_ld=(--diff_ld_ckpt "$ld_ck" --diff_ld_draws 10 --diff_ld_cg_iters 3)
      fi
      run_job "${tag}_3c391_${mname}" "$suite_dir/3c391_${mname}" \
        scripts/run_imaging.py \
        --vis "$MS_3C391" --imagename 3c391 --field '3C391*' \
        --out_dir "$suite_dir/3c391_${mname}" \
        --imsize 480 --cell 2.5arcsec --gridder mosaic --wprojplanes -1 --specmode mfs --robust 0.5 --pblimit -0.05 \
        --mask "$MASK_3C391" --mask_radius_px 200 --n_sigma 3.0 --no_psf_floor --global_threshold 0.001 \
        --loop_gain 0.2 --max_major 12 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
        --diffusion_ckpt "$p_ck" --diff_mp --diff_n_draws "$n_b" --diff_seed 0 "${extra_ld[@]}" \
        --fit_unmasked --diff_unmask_output --diff_taper --diff_q_space --diff_dof_pen 2.0 --diff_briggs_psd \
        --refit_exact --major_before_refit --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
        --diff_rebuild_drop 0.2 --diff_save_atoms --cycle_report \
        --device "$DEVICE"
    done

    # Unmasked 3C391 test for the AIC DOF speckle fix
    run_job "${tag}_3c391_B2_unmasked" "$suite_dir/3c391_B2_unmasked" \
      scripts/run_imaging.py \
      --vis "$MS_3C391" --imagename 3c391 --field '3C391*' \
      --out_dir "$suite_dir/3c391_B2_unmasked" \
      --imsize 480 --cell 2.5arcsec --gridder mosaic --wprojplanes -1 --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask_radius_px 0 --n_sigma 3.0 --no_psf_floor --global_threshold 0.001 \
      --loop_gain 0.2 --max_major 12 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --diffusion_ckpt "$diff_p" --diff_mp --diff_n_draws 10 --diff_seed 0 \
      --fit_unmasked --diff_unmask_output --diff_taper --diff_q_space --diff_dof_pen 2.0 --diff_briggs_psd \
      --refit_exact --major_before_refit --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
      --diff_rebuild_drop 0.2 --diff_save_atoms \
      --device "$DEVICE"
  fi

  # 3. G55.7+3.4 (1280 px, 8 arcsec, wproject 128, unmasked)
  if [[ -d "$MS_G55" ]]; then
    run_job "${tag}_g55_mdn" "$suite_dir/g55_mdn" \
      scripts/run_imaging.py \
      --vis "$MS_G55" --imagename g55 --out_dir "$suite_dir/g55_mdn" \
      --imsize 1280 --cell 8arcsec --gridder wproject --wprojplanes 128 --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask_radius_px 0 --n_sigma 3 --no_psf_floor --global_sigma \
      --loop_gain 0.1 --max_major 10 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --ckpt "$mdn_128" --sigma_min_px 0.5 --sigma_max_px 96.0 --fit_unmasked \
      --mdn_density_mode --mdn_profile_amp --mdn_delta_fallback --uv_coupling_min 0.2 \
      --refit_exact --major_before_refit --refit_all --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
      --device "$DEVICE"

    for mode_pair in "B2|$diff_p||10" "A3|$diff_p|$diff_ld|0" "B2_A3|$diff_p|$diff_ld|10"; do
      IFS='|' read -r mname p_ck ld_ck n_b <<< "$mode_pair"
      extra_ld=()
      if [[ -n "$ld_ck" ]]; then
        extra_ld=(--diff_ld_ckpt "$ld_ck" --diff_ld_draws 10 --diff_ld_cg_iters 3)
      fi
      run_job "${tag}_g55_${mname}" "$suite_dir/g55_${mname}" \
        scripts/run_imaging.py \
        --vis "$MS_G55" --imagename g55 --out_dir "$suite_dir/g55_${mname}" \
        --imsize 1280 --cell 8arcsec --gridder wproject --wprojplanes 128 --specmode mfs --robust 0.5 --pblimit -0.05 \
        --mask_radius_px 0 --n_sigma 3 --no_psf_floor --global_sigma \
        --loop_gain 0.2 --max_major 10 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
        --diffusion_ckpt "$p_ck" --diff_mp --diff_n_draws "$n_b" --diff_seed 0 "${extra_ld[@]}" \
        --fit_unmasked --diff_unmask_output --diff_taper --diff_q_space --diff_dof_pen 2.0 --diff_briggs_psd \
        --refit_exact --major_before_refit --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
        --diff_rebuild_drop 0.2 --diff_save_atoms \
        --device "$DEVICE"
    done
  fi

  # 4. Cygnus A (2048 px, 0.2 arcsec, standard, unmasked, multi-scale 16,32,64 for sharp hotspots)
  if [[ -d "$MS_CYGA" ]]; then
    run_job "${tag}_cygA_mdn" "$suite_dir/cygA_mdn" \
      scripts/run_imaging.py \
      --vis "$MS_CYGA" --imagename cygA --out_dir "$suite_dir/cygA_mdn" \
      --imsize 2048 --cell 0.2arcsec --gridder standard --wprojplanes -1 --specmode mfs --robust 0.5 --pblimit -0.05 \
      --mask_radius_px 0 --n_sigma 1 --no_psf_floor --global_sigma --sidelobe_level 0.06 \
      --loop_gain 0.1 --max_major 15 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
      --ckpt "$mdn_256" --sigma_min_px 0.5 --sigma_max_px 96.0 --fit_unmasked \
      --mdn_density_mode --mdn_profile_amp --mdn_delta_fallback --uv_coupling_min 0.2 \
      --refit_exact --major_before_refit --refit_all --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
      --device "$DEVICE"

    for mode_pair in "B2|$diff_p||10" "A3|$diff_p|$diff_ld|0" "B2_A3|$diff_p|$diff_ld|10"; do
      IFS='|' read -r mname p_ck ld_ck n_b <<< "$mode_pair"
      extra_ld=()
      if [[ -n "$ld_ck" ]]; then
        extra_ld=(--diff_ld_ckpt "$ld_ck" --diff_ld_draws 10 --diff_ld_cg_iters 3)
      fi
      run_job "${tag}_cygA_${mname}" "$suite_dir/cygA_${mname}" \
        scripts/run_imaging.py \
        --vis "$MS_CYGA" --imagename cygA --out_dir "$suite_dir/cygA_${mname}" \
        --imsize 2048 --cell 0.2arcsec --gridder standard --wprojplanes -1 --specmode mfs --robust 0.5 --pblimit -0.05 \
        --mask_radius_px 0 --n_sigma 1 --no_psf_floor --global_sigma --sidelobe_level 0.06 \
        --loop_gain 0.2 --max_major 15 --max_components 5000 --divergence_tol 0.2 --config_idx 3 \
        --diffusion_ckpt "$p_ck" --diff_mp --diff_n_draws "$n_b" --diff_seed 0 "${extra_ld[@]}" \
        --diff_scales 16,32,64 \
        --fit_unmasked --diff_unmask_output --diff_taper --diff_q_space --diff_dof_pen 2.0 --diff_briggs_psd \
        --refit_exact --major_before_refit --refit_method nnls --refit_gain 0.5 --refit_nullspace_lam 1e-5 \
        --diff_rebuild_drop 0.2 --diff_save_atoms \
        --device "$DEVICE"
    done
  fi
}

tier1_eval() {
  log_status "TIER1_EVAL starting"
  run_imaging_suite "tier1_v1ckpts" "$CKPT_MDN_V1" "$CKPT_MDN_256_V1" "$CKPT_B2_V1" "$CKPT_A3_V1"
  log_status "TIER1_EVAL completed"
}

retrain_all() {
  log_status "RETRAIN_ALL starting"
  local ckpt_root="$OUT_ROOT/checkpoints_v2"
  mkdir -p "$ckpt_root"

  local corpus_arg=()
  if [[ -d "$CORPUS_PSF_DIR" ]]; then
    corpus_arg=(--corpus_psf_dir "$CORPUS_PSF_DIR")
  fi

  # 1. MDN-Asp 128 px (6D von Mises PA + uniform point PA + correlated PSF noise)
  run_job "train_mdn_128_v2" "$ckpt_root/mdn_128_v2" \
    scripts/train_mdn_asp.py \
    --out_dir "$ckpt_root/mdn_128_v2" --repo_root "$PSF_ROOT" "${corpus_arg[@]}" \
    --steps 50000 --batch_size 128 --lr 1e-3 --val_size 1024 \
    --morphologies point,blob,amorphous --extended_fraction 0.05 --snr_min 5.0 \
    --scale_relative --correlated_noise --point_sigma_px 0.5 --blob_sigma_min_px 0.5 \
    --cutout 128 --pa_head von_mises --uniform_point_pa \
    --seed 42 --device "$DEVICE"

  # 2. MDN-Asp 256 px (for Cygnus A)
  run_job "train_mdn_256_v2" "$ckpt_root/mdn_256_v2" \
    scripts/train_mdn_asp.py \
    --out_dir "$ckpt_root/mdn_256_v2" --repo_root "$PSF_ROOT" "${corpus_arg[@]}" \
    --steps 50000 --batch_size 64 --lr 1e-3 --val_size 1024 \
    --morphologies point,blob,amorphous --extended_fraction 0.05 --snr_min 5.0 \
    --scale_relative --correlated_noise --point_sigma_px 0.5 --blob_sigma_min_px 0.5 \
    --cutout 256 --pa_head von_mises --uniform_point_pa \
    --seed 42 --device "$DEVICE"

  # 3. Shape-prior diffusion P v2 (bf16 enabled for A100 Tensor Cores)
  run_job "train_diff_P_v2" "$ckpt_root/diff_P_v2" \
    scripts/train_tiled_asp.py \
    --out_dir "$ckpt_root/diff_P_v2" --variant P \
    --repo_root "$PSF_ROOT" "${corpus_arg[@]}" --agn_npz "$AGN_NPZ" \
    --shape_size 64 --patch 256 --gamma 2.0 --crop_floor 0.1 \
    --steps 50000 --batch_size 128 --lr 2e-4 --val_size 128 \
    --bf16 --seed 42 --device "$DEVICE"

  # 4. Conditional-dictionary diffusion LD v2 (learned conv context stem + partial-clean residuals + AGN distractors + bf16)
  run_job "train_diff_LD_v2" "$ckpt_root/diff_LD_v2" \
    scripts/train_tiled_asp.py \
    --out_dir "$ckpt_root/diff_LD_v2" --variant LD \
    --repo_root "$PSF_ROOT" "${corpus_arg[@]}" --agn_npz "$AGN_NPZ" \
    --ctx 256 --shape_size 64 --patch 512 --psf_size 512 --gamma 2.0 \
    --n_extra 4 --agn_mean 25.16 --snr_range 1.0 1000.0 \
    --ld_ctx_encoder conv --ld_clean_prob 0.5 \
    --steps 50000 --batch_size 64 --lr 2e-4 --val_size 128 \
    --bf16 --seed 42 --device "$DEVICE"

  # Export safetensors HF directories immediately
  local hf_v2="$ckpt_root/hf_export"
  run_job "hf_mdn_128_v2" "$hf_v2/mdn_asp_128" \
    scripts/convert_checkpoint_to_hf.py --ckpt "$ckpt_root/mdn_128_v2/best.pt" --out_dir "$hf_v2/mdn_asp_128"
  run_job "hf_mdn_256_v2" "$hf_v2/mdn_asp_256" \
    scripts/convert_checkpoint_to_hf.py --ckpt "$ckpt_root/mdn_256_v2/best.pt" --out_dir "$hf_v2/mdn_asp_256"
  run_job "hf_diff_P_v2" "$hf_v2/diffusion_B2_v2" \
    scripts/convert_checkpoint_to_hf.py --diffusion --ckpt "$ckpt_root/diff_P_v2/best.pt" --out_dir "$hf_v2/diffusion_B2_v2"
  run_job "hf_diff_LD_v2" "$hf_v2/diffusion_A3_v2" \
    scripts/convert_checkpoint_to_hf.py --diffusion --ckpt "$ckpt_root/diff_LD_v2/best.pt" --out_dir "$hf_v2/diffusion_A3_v2"

  log_status "RETRAIN_ALL completed"
}

tier2_eval() {
  log_status "TIER2_EVAL starting"
  local ckpt_root="$OUT_ROOT/checkpoints_v2"
  local mdn_128="$ckpt_root/mdn_128_v2/best.pt"
  local mdn_256="$ckpt_root/mdn_256_v2/best.pt"
  local diff_p="$ckpt_root/diff_P_v2/best.pt"
  local diff_ld="$ckpt_root/diff_LD_v2/best.pt"

  # Offline held-out diffusion & sim_recovery gates
  run_job "eval_diff_P_v2" "$OUT_ROOT/tier2_v2ckpts/eval_diff_P" \
    scripts/eval_diffusion.py --ckpt "$diff_p" --out_dir "$OUT_ROOT/tier2_v2ckpts/eval_diff_P" --device "$DEVICE"
  run_job "eval_diff_LD_v2" "$OUT_ROOT/tier2_v2ckpts/eval_diff_LD" \
    scripts/eval_diffusion.py --ckpt "$diff_ld" --out_dir "$OUT_ROOT/tier2_v2ckpts/eval_diff_LD" --device "$DEVICE"

  if [[ -f "$PSF_ROOT/data/g55/chunk_00/psf.fits" ]]; then
    run_job "sim_recovery_mdn_v2" "$OUT_ROOT/tier2_v2ckpts/sim_recovery_mdn" \
      scripts/sim_recovery.py \
      --ckpt "$mdn_128" --psf "$PSF_ROOT/data/g55/chunk_00/psf.fits" \
      --out_dir "$OUT_ROOT/tier2_v2ckpts/sim_recovery_mdn" \
      --refit_exact --refit_method nnls --refit_nullspace_lam 1e-4 \
      --mdn_density_mode --mdn_profile_amp --mdn_delta_fallback --uv_coupling_min 0.2 \
      --device "$DEVICE"
  fi

  run_imaging_suite "tier2_v2ckpts" "$mdn_128" "$mdn_256" "$diff_p" "$diff_ld"
  log_status "TIER2_EVAL completed"
}

eval_jet() {
  local suite_dir="${1:-$OUT_ROOT/tier1_v1ckpts}"
  if [[ ! -d "$SIM_JET_DIR/sim_jet_vla.im" ]]; then
    echo "[eval_jet] Skipping: $SIM_JET_DIR/sim_jet_vla.im not found"
    return 0
  fi
  (
    cd "$REPO_DIR"
    "${PY_CMD[@]}" - "$SIM_JET_DIR" "$suite_dir" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from casatools import image as iatool

sim_dir = Path(sys.argv[1])
suite_dir = Path(sys.argv[2])

def read_im(p):
    ia = iatool()
    ia.open(str(p))
    arr = ia.getchunk().squeeze().T.astype(np.float64)
    ia.close()
    return arr

truth = read_im(sim_dir / "sim_jet_vla.im")
sup_raw = truth > 1e-6 * truth.max()
sup_dil = binary_dilation(sup_raw, iterations=10)

# Locate true core pixel (brightest compact peak in truth)
cy, cx = np.unravel_index(int(np.argmax(truth)), truth.shape)
core_box = np.zeros_like(sup_raw)
core_box[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6] = True

candidates = []
for name, p in [
    ("MDN-Asp (new)", suite_dir / "sim_jet_mdn"),
    ("Diffusion B2 (new)", suite_dir / "sim_jet_B2"),
    ("MDN-Asp (v1 baseline)", sim_dir / "mdn_asp_smax11p2_thr0.05"),
    ("Diffusion B2 (v1 baseline)", sim_dir / "mp_B2_thr0.05"),
    ("CASA Asp", sim_dir / "casa_asp"),
]:
    if (p / "jet.model").exists():
        candidates.append((name, p))

rows = []
true_tot = float(truth.sum())
true_sup = float(truth[sup_dil].sum())
true_core = float(truth[core_box].sum())

for name, p in candidates:
    mod = read_im(p / "jet.model")
    res = read_im(p / "jet.residual")
    rows.append({
        "run": name,
        "dir": str(p),
        "total_jy": float(mod.sum()),
        "total_ratio": float(mod.sum() / true_tot),
        "on_support_jy": float(mod[sup_dil].sum()),
        "on_support_ratio": float(mod[sup_dil].sum() / true_sup),
        "off_support_jy": float(mod[~sup_dil].sum()),
        "core_11x11_jy": float(mod[core_box].sum()),
        "core_11x11_ratio": float(mod[core_box].sum() / true_core),
        "residual_rms": float(np.sqrt(np.mean(res ** 2))),
    })

out_json = suite_dir / "sim_jet_summary.json"
out_json.write_text(json.dumps({"truth_total_jy": true_tot, "truth_core_11x11_jy": true_core, "runs": rows}, indent=2))
lines = [
    "| Run | Total (Jy) | Total/True | On-support (Jy) | On/True | Off-support (Jy) | Core 11x11/True | Resid RMS (Jy/beam) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['run']} | {r['total_jy']:.3f} | {r['total_ratio']:.3f} | "
        f"{r['on_support_jy']:.3f} | {r['on_support_ratio']:.3f} | "
        f"{r['off_support_jy']:.3f} | {r['core_11x11_ratio']:.3f} | {r['residual_rms']:.4e} |"
    )
out_md = suite_dir / "sim_jet_summary.md"
out_md.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
  )
}

cmd="${1:-preflight}"
case "$cmd" in
  preflight)   preflight ;;
  tier1_eval)  preflight && tier1_eval ;;
  retrain_all) preflight && retrain_all ;;
  tier2_eval)  preflight && tier2_eval ;;
  eval_jet)    eval_jet "${2:-$OUT_ROOT/tier1_v1ckpts}" ;;
  all)         preflight && tier1_eval && retrain_all && tier2_eval ;;
  *)
    echo "Usage: $0 {preflight|tier1_eval|retrain_all|tier2_eval|eval_jet|all}" >&2
    exit 1
    ;;
esac
