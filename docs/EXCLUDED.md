Excluded from this release, and why
----------------------------------
SSH and file-transfer helpers
    fetch_server_results.py, fetch_server_scripts.py, probe_remote_paths.py,
    verify_sht_split.py, remote.py
    These exist only to move files off the analysis host and embed its address
    and account name. They are operator infrastructure rather than part of the
    method. verify_sht_split.py needs them, so its output is already captured in
    results/sht_split_verified.json.

Windows launcher scripts
    g*_run.bat, p3_*.bat, py.bat, run_py.bat, s0_*.bat, s2b_run.bat
    They hardcode absolute paths to start runs on one machine. Every script they
    launch already takes its paths as command-line arguments, so the same run is
    reproducible from the documented invocations instead.

Score dumps for UCF-Crime and XD-Violence
    The archived per-clip scores for the two mainstream benchmarks are large and
    fully regenerable from the released checkpoints. The ShanghaiTech archives
    are kept because that benchmark carries the sign-reversal claim.

ShanghaiTech frame masks
    Redistributed by the ShanghaiTech dataset itself. Mirroring them here would
    duplicate someone else's release; point at the dataset instead.
