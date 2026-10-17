"""Upload a script to the server, run it there, stream back its output.

Two reasons not to compute this locally. The UCF I3D features are 2.2 GB across
290 files, and transferring them over SFTP to inspect 290 integers is
indefensible when the files already sit on a machine with a GPU. More
importantly, the .npy header parser in this environment's numpy cannot read
these files, while ``np.load`` on a full buffer can -- so a local header-read
approach fails for reasons unrelated to the data.

The server also runs its own numpy, which may not share that quirk. This script
therefore reports the header-parse result separately from np.load rather than
assuming they agree.

Usage: run_on_server.py <local_script.py> [remote_name]
"""

from __future__ import annotations

import os
import posixpath
import sys

import paramiko

# Read from the environment rather than hard-coded. The address and account name
# identify an internal machine and are not something a public repository should
# carry; the scripts work unchanged once the two variables are exported.
HOST = os.environ.get("CDV_SSH_HOST", "")
USER = os.environ.get("CDV_SSH_USER", "")
if not HOST or not USER:
    sys.stderr.write("CDV_SSH_HOST and CDV_SSH_USER must be set; this script "
                     "does not carry a default host.\n")
    raise SystemExit(2)
REMOTE_DIR = os.environ.get("CDV_REMOTE_DIR", "")
if not REMOTE_DIR:
    sys.stderr.write("CDV_REMOTE_DIR must be set to the directory the remote "
                     "scripts are staged in.\n")
    raise SystemExit(2)
# Long-running feature work needs more than the default 5 s channel timeout.
CHANNEL_TIMEOUT_S = 3600


def get_password() -> str:
    pw = os.environ.get("CDV_SSH_PW")
    if not pw:
        sys.stderr.write("CDV_SSH_PW is not set; refusing to connect.\n")
        raise SystemExit(2)
    return pw


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: run_on_server.py <local_script.py> [remote_name]\n")
        return 2
    local = sys.argv[1]
    if not os.path.isfile(local):
        sys.stderr.write(f"no such local file: {local}\n")
        return 2
    name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(local)
    forwarded = sys.argv[3:]
    remote = posixpath.join(REMOTE_DIR, name).replace("\\", "/")
    # The server is Windows, so the remote path must keep backslashes for the
    # shell even though SFTP wants forward slashes.
    remote_native = remote.replace("/", "\\")

    with open(local, "rb") as fh:
        payload = fh.read()

    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        cli.connect(HOST, username=USER, password=get_password(), timeout=20,
                    look_for_keys=False, allow_agent=False)
    except paramiko.SSHException as exc:
        sys.stderr.write(f"connection failed: {type(exc).__name__}: {exc}\n")
        return 3

    print(f"uploading {local} ({len(payload)} bytes) -> {remote_native}")
    # The remote directory may not exist on a fresh server; creating it is the
    # only write this tool performs, and it is idempotent.
    parent = os.path.dirname(remote_native)
    _si, _so, _se = cli.exec_command(f'mkdir "{parent}" 2>nul')
    _so.channel.recv_exit_status()
    sftp = cli.open_sftp()
    with sftp.open(remote_native, "wb") as fh:
        fh.write(payload)
    sftp.close()

    cmd = f'python "{remote_native}"'
    if forwarded:
        cmd += " " + " ".join(f'"{a}"' for a in forwarded)
    # CUDA_LAUNCH_BLOCKING and friends must be set before the CUDA context is
    # created, so they are exported by the shell rather than inside the script.
    remote_env = os.environ.get("CDV_REMOTE_ENV", "").strip()
    if remote_env:
        cmd = "set " + remote_env + " && " + cmd
    print(f"running: {cmd}\n")
    _stdin, stdout, stderr = cli.exec_command(cmd, timeout=CHANNEL_TIMEOUT_S)
    stdout.channel.settimeout(CHANNEL_TIMEOUT_S)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()

    if out:
        print(out.rstrip())
    if err.strip():
        print("\n--- stderr ---", file=sys.stderr)
        print(err.rstrip(), file=sys.stderr)
    print(f"\n(exit {code})")

    cli.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
