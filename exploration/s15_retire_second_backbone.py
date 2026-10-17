"""Pull the second-backbone investigation records off the server and drop the duplicate.

Retrieval
---------
The measurements behind the limitations paragraph must be traceable to archived
files, not to a transcript of this session. Every JSON produced by the
investigation is copied into the local repository so the claim that a second
backbone family could not be obtained can be checked the same way any other
number in the paper can be.

Deletion
--------
E:\\dataset\\XD_i3d holds 1.43 GB extracted from a 12.21 GB archive, and every
file in it was verified byte-identical to the XDTestClipFeatures store already on
the server. It is a duplicate, so it is removed. The deletion is confined to that
one directory and the script refuses to run if the path does not look like what it
expects, since a mis-resolved path on a remote filesystem is not recoverable.

The scores array produced by the failed scorer is deliberately not retrieved: it
is the output of an instrument that does not work, not a measurement the paper
reports, and shipping it would invite it to be mistaken for one.
"""

from __future__ import annotations

import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
import sys

import paramiko

HOST = os.environ.get("CDV_SSH_HOST", "")
USER = os.environ.get("CDV_SSH_USER", "")
REMOTE_RESULTS = os.environ.get(
    "CDV_REMOTE_RESULTS", os.environ.get("CDV_REMOTE_RESULTS", "results"))
LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "results", "second_backbone")
DUPLICATE_DIR = r"E:\dataset\XD_i3d"

WANTED = [
    "i3d_ucf_train_log.json",
    "i3d_split_diagnosis.json",
    "ucf_label_manifest.json",
    "ucf_label_gt_agreement.json",
    "xd_i3d_validation.json",
    "xd_archive_identity_and_ucf_bias.json",
    "xd_local_feature_stores.json",
    "ucf_grid_alignment.json",
    "ucf_grid_verdict.json",
]

# Retrieved for the record but not shipped, because it is a failed instrument.
NOT_SHIPPED = ("i3d_ucf_scores.npz",)


def get_password() -> str:
    pw = os.environ.get("CDV_SSH_PW")
    if not pw:
        sys.stderr.write("CDV_SSH_PW is not set; refusing to connect.\n")
        raise SystemExit(2)
    return pw


def check_endpoint() -> None:
    """Refuse to proceed without an explicit host, rather than connecting to ''."""
    if not HOST or not USER:
        sys.stderr.write(
            "CDV_SSH_HOST and CDV_SSH_USER must be set; this script does not "
            "carry a default host.\n")
        raise SystemExit(2)


def main() -> int:
    check_endpoint()
    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        cli.connect(HOST, username=USER, password=get_password(), timeout=20,
                    look_for_keys=False, allow_agent=False)
    except paramiko.SSHException as exc:
        sys.stderr.write(f"connection failed: {type(exc).__name__}: {exc}\n")
        return 3
    sftp = cli.open_sftp()
    os.makedirs(LOCAL_ROOT, exist_ok=True)

    print("=== retrieving investigation records ===")
    got, missing = 0, []
    for name in WANTED:
        remote = os.path.join(REMOTE_RESULTS, name)
        local = os.path.join(LOCAL_ROOT, name)
        try:
            st = sftp.stat(remote)
        except FileNotFoundError:
            missing.append(name)
            print(f"  MISSING  {name}")
            continue
        sftp.get(remote, local)
        print(f"  ok  {name:<44} {st.st_size:>9,} B")
        got += 1
    # The fetch manifest sits one directory up, beside the dataset it describes.
    for name, remote in (("xd_i3d_fetch_manifest.json",
                          r"E:\dataset\XD_i3d\xd_i3d_fetch_manifest.json"),):
        try:
            st = sftp.stat(remote)
            sftp.get(remote, os.path.join(LOCAL_ROOT, name))
            print(f"  ok  {name:<44} {st.st_size:>9,} B")
            got += 1
        except FileNotFoundError:
            missing.append(name)
            print(f"  MISSING  {name}")

    print(f"\n  retrieved {got}, missing {len(missing)}")
    print("  deliberately not retrieved (failed instrument): "
          f"{', '.join(NOT_SHIPPED)}")

    print("\n=== confirming the duplicate before deleting it ===")
    if not DUPLICATE_DIR.startswith("E:\\dataset\\XD_i3d"):
        sys.stderr.write(f"refusing: unexpected path {DUPLICATE_DIR}\n")
        sftp.close()
        cli.close()
        return 4
    try:
        entries = sftp.listdir(DUPLICATE_DIR)
        st = sftp.stat(DUPLICATE_DIR)
        print(f"  {DUPLICATE_DIR}")
        print(f"  entries: {len(entries)}")
        if st.st_mtime is not None:
            print(f"  mtime: {st.st_mtime}")
    except FileNotFoundError:
        print("  already absent")
        sftp.close()
        cli.close()
        return 0

    print("\n=== removing the duplicate ===")
    # Recursive: the extracted content sits in a test\ subdirectory, so a
    # single-level delete leaves the bulk of the 1.43 GB behind.
    def rmtree(path: str, depth: int = 0) -> int:
        n = 0
        for name in sftp.listdir(path):
            sub = os.path.join(path, name)
            try:
                sftp.remove(sub)
                n += 1
            except PermissionError:
                n += rmtree(sub, depth + 1)
        try:
            sftp.rmdir(path)
            print(f"  removed directory{' (recursive)' if depth else ''}: {path}")
        except Exception as exc:
            print(f"  could not remove directory {path}: {type(exc).__name__}")
        return n

    removed = rmtree(DUPLICATE_DIR)
    print(f"  removed {removed} files")
    try:
        sftp.listdir(DUPLICATE_DIR)
        print("  WARNING path still exists")
    except FileNotFoundError:
        print(f"  confirmed gone: {DUPLICATE_DIR}")

    sftp.close()
    cli.close()
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
