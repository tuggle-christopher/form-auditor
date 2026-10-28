"""Reading inputs from / writing results to cloud storage (Databricks, Azure Blob).

The auditor itself works on local files. Remote locations are copied to a local temp
folder first and the results are copied back afterwards:

  /dbfs/mnt/..., /Volumes/..., any local path   used directly (no copying)
  abfss://container@account.dfs.core.windows.net/path
  wasbs://container@account.blob.core.windows.net/path
  dbfs:/mnt/...                                 copied with Databricks `dbutils.fs.cp`, using
                                                the storage access configured on the cluster
  https://account.blob.core.windows.net/container/path
                                                copied with the azure-storage-blob package,
                                                using credentials from the environment / .env:
                                                AZURE_STORAGE_CONNECTION_STRING, or
                                                AZURE_STORAGE_SAS_TOKEN, or AZURE_STORAGE_KEY
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .io_utils import IMAGE_EXTS, PDF_EXTS

_DBUTILS_SCHEMES = ("abfss://", "abfs://", "wasbs://", "wasb://", "dbfs:/")
_HTTPS_BLOB = re.compile(r"^https://([^./]+)\.blob\.core\.windows\.net/([^/]+)/?(.*)$")
INPUT_EXTS = IMAGE_EXTS | PDF_EXTS


def is_remote(p: str | Path) -> bool:
    s = str(p)
    return s.startswith(_DBUTILS_SCHEMES) or bool(_HTTPS_BLOB.match(s))


def _ext_ok(name: str) -> bool:
    return Path(name.rstrip("/")).suffix.lower() in INPUT_EXTS


# --------------------------------------------------------------------------- #
# Databricks dbutils
# --------------------------------------------------------------------------- #
_dbutils = None


def set_dbutils(dbutils) -> None:
    """Pass the notebook's `dbutils` explicitly (optional - it is found automatically)."""
    global _dbutils
    _dbutils = dbutils


def get_dbutils():
    global _dbutils
    if _dbutils is not None:
        return _dbutils
    try:                                            # inside a notebook
        import IPython
        ip = IPython.get_ipython()
        if ip is not None and "dbutils" in ip.user_ns:
            _dbutils = ip.user_ns["dbutils"]
            return _dbutils
    except Exception:
        pass
    try:                                            # jobs / python files on a cluster
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        _dbutils = DBUtils(SparkSession.builder.getOrCreate())
        return _dbutils
    except Exception as e:
        raise RuntimeError(
            "abfss:// / wasbs:// / dbfs:/ paths need Databricks dbutils, which was not found. "
            "Run on a Databricks cluster, call form_auditor.storage.set_dbutils(dbutils), or "
            "use a local path such as /dbfs/mnt/...") from e


def _db_download(uri: str, local_dir: Path) -> list[Path]:
    du = get_dbutils()
    local_dir.mkdir(parents=True, exist_ok=True)
    if _ext_ok(uri):                                   # a single file
        pairs = [(uri, uri.rstrip("/").rsplit("/", 1)[-1])]
    else:                                              # a folder (not recursive)
        try:
            entries = du.fs.ls(uri)
        except Exception as e:
            raise FileNotFoundError(f"{uri}: {e}") from e
        pairs = [(e.path, e.name) for e in entries if not e.name.endswith("/") and _ext_ok(e.name)]
    out = []
    for src, name in sorted(pairs, key=lambda p: p[1]):
        dest = local_dir / name
        du.fs.cp(src, "file:" + str(dest))
        out.append(dest)
    return out


def _db_upload(local_dir: Path, uri: str) -> None:
    du = get_dbutils()
    du.fs.mkdirs(uri)
    du.fs.cp("file:" + str(local_dir), uri, recurse=True)


# --------------------------------------------------------------------------- #
# Azure Blob over https (azure-storage-blob)
# --------------------------------------------------------------------------- #
def _container_client(account: str, container: str):
    try:
        from azure.storage.blob import ContainerClient
    except ImportError as e:
        raise RuntimeError("https://<account>.blob.core.windows.net paths need the "
                           "azure-storage-blob package: pip install azure-storage-blob") from e
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        return ContainerClient.from_connection_string(conn, container)
    url = f"https://{account}.blob.core.windows.net"
    sas = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    if sas:
        return ContainerClient(url, container, credential=sas.lstrip("?"))
    key = os.environ.get("AZURE_STORAGE_KEY")
    if key:
        return ContainerClient(url, container, credential={"account_name": account, "account_key": key})
    raise RuntimeError("No blob storage credentials: set AZURE_STORAGE_CONNECTION_STRING, "
                       "AZURE_STORAGE_SAS_TOKEN or AZURE_STORAGE_KEY (e.g. in .env)")


def _blob_download(url: str, local_dir: Path) -> list[Path]:
    account, container, prefix = _HTTPS_BLOB.match(url).groups()
    cc = _container_client(account, container)
    local_dir.mkdir(parents=True, exist_ok=True)
    names = []
    if prefix and _ext_ok(prefix):
        names = [prefix]
    else:
        pre = prefix.rstrip("/") + "/" if prefix else ""
        names = [b.name for b in cc.list_blobs(name_starts_with=pre)
                 if "/" not in b.name[len(pre):] and _ext_ok(b.name)]     # not recursive
    out = []
    for n in sorted(names):
        dest = local_dir / Path(n).name
        with open(dest, "wb") as fh:
            cc.download_blob(n).readinto(fh)
        out.append(dest)
    return out


def _blob_upload(local_dir: Path, url: str) -> None:
    account, container, prefix = _HTTPS_BLOB.match(url).groups()
    cc = _container_client(account, container)
    pre = prefix.rstrip("/") + "/" if prefix else ""
    for f in sorted(local_dir.rglob("*")):
        if f.is_file():
            with open(f, "rb") as fh:
                cc.upload_blob(pre + f.relative_to(local_dir).as_posix(), fh, overwrite=True)


# --------------------------------------------------------------------------- #
def download(uri: str, local_dir: Path) -> list[Path]:
    """Copy a remote file, or the supported files directly inside a remote folder."""
    files = _blob_download(uri, local_dir) if uri.startswith("https://") else _db_download(uri, local_dir)
    if not files:
        raise FileNotFoundError(f"No PDF / image files found at {uri}")
    return files


def upload(local_dir: Path, uri: str) -> None:
    """Copy a local folder's contents into the remote folder `uri`."""
    if uri.startswith("https://"):
        _blob_upload(local_dir, uri)
    else:
        _db_upload(local_dir, uri)


def stage_inputs(inputs: list[str | Path], tmp: Path) -> list[Path]:
    """Local paths for all inputs (remote ones are downloaded into `tmp`)."""
    out = []
    for i, p in enumerate(inputs):
        if is_remote(p):
            out += download(str(p), tmp / f"in{i}")
        else:
            out.append(Path(p))
    return out


def local_output_dir(out: str | Path, tmp: Path) -> tuple[Path, Optional[str]]:
    """-> (local folder to write to, remote destination or None)."""
    if is_remote(out):
        return tmp / "out", str(out)
    return Path(out), None
