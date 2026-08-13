# Active continuous base-Y tracking backup

This preserved source snapshot is selected only for explicit continuous mouth
tracking requests. The guarded real-backend router launches it through
`scripts/real_feed_water_integrated_base_y_backup.py`, which loads the backup
controller, tracker, runner, and configuration in an isolated child process.

The established camera-ray implementation remains byte-for-byte untouched and
is not selected for continuous tracking. Frozen-target and legacy segmented
modes continue to use their established runner. Do not run backup files
directly; use the canonical guarded Codex entry point so all authorization and
runtime gates remain active.
