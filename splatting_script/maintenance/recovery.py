"""Verify, list and selectively restore sealed research-source archives.

Shared recovery utility; archived implementations are never imported.
Main API: records, verify, safe_path, main."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import os

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def safe_path(base, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or p.parts[0] == '.git':
        raise ValueError('Unsafe recovery member: '+name)
    target = base/p
    if not target.parent.resolve().is_relative_to(base.resolve()):
        raise ValueError('Recovery path leaves destination: '+name)
    return target


def records(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT/'archive/three_backends'):
        raise ValueError('Expected an in-repository consolidation archive')
    seal = json.loads((directory/'seal.json').read_text())
    for name in ('manifest.json', 'retired.tar.gz'):
        if digest(directory/name) != seal[name]['sha256']:
            raise ValueError('Archive seal mismatch: '+name)
    return {row['path']:row for row in json.loads((directory/'manifest.json').read_text())['files']}


def verify(directory, selected=(), destination=None):
    directory = Path(directory).resolve()
    expected = records(directory)
    selection = set(selected)
    if not selection <= expected.keys(): raise ValueError('Unknown selected recovery file')
    if destination is not None:
        destination = Path(destination).resolve()
        if not destination.is_relative_to(ROOT): raise ValueError('Recovery destination must be inside repository')
        if not selection: raise ValueError('Explicit file selection required for restoration')
        # Preflight every destination before creating the first file.
        for name in selection:
            target = safe_path(destination, name)
            if target.exists() or target.is_symlink(): raise FileExistsError(target)
    seen = set()
    with tarfile.open(directory/'retired.tar.gz', 'r:gz') as archive:
        for member in archive:
            name = member.name
            safe_path(ROOT, name)
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError('Unexpected, nonregular or duplicate archive member: '+name)
            seen.add(name)
            row = expected[name]
            if member.size != row['bytes']: raise ValueError('Member size mismatch: '+name)
            h = hashlib.sha256()
            source = archive.extractfile(member)
            target = None
            if destination is not None and name in selection:
                target = safe_path(destination, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('xb')
            else: output = None
            try:
                for block in iter(lambda: source.read(1024*1024), b''):
                    h.update(block)
                    if output: output.write(block)
            finally:
                if output: output.close()
                source.close()
            if h.hexdigest() != row['sha256']: raise ValueError('Member hash mismatch: '+name)
            if target:
                target.chmod(row['mode'])
                os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
    if seen != expected.keys(): raise ValueError('Archive has missing members')
    return dict(verified_files=len(seen), restored_files=len(selection) if destination is not None else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--restore', nargs='+', metavar='REPOSITORY_RELATIVE_FILE')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(records(args.archive)))
    else:
        print(json.dumps(verify(args.archive, args.restore or (), ROOT if args.restore else None)))


if __name__ == '__main__': main()
