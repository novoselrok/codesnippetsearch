import os
import tempfile
import subprocess


def get_tmp_repository_dir_path(tmp_dir: tempfile.TemporaryDirectory, organization: str, name: str):
    return os.path.join(tmp_dir.name, organization, name)


def download_repository(organization: str, name: str, target_dir):
    os.environ['GIT_TERMINAL_PROMPT'] = '0'
    cmd = [
        'git',
        'clone',
        '--depth=1',
        'https://github.com/{}/{}.git'.format(organization, name),
        target_dir
    ]
    subprocess.run(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def get_repository_commit_hash(repository_dir):
    cwd = os.getcwd()
    os.chdir(repository_dir)
    # git rev-parse HEAD
    cmd = ['git', 'rev-parse', 'HEAD']
    sha = subprocess.check_output(cmd).strip().decode('utf-8')
    os.chdir(cwd)
    return sha
