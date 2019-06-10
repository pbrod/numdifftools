"""
Script to build the numdifftools package.

The script remove the previous built binaries and generated documentation
before it generate the documentation and build the binaries and finally
check the built binaries.

It assumes that the numdifftools library is installed in so called develop mode.

Created on 7. des. 2018

@author: pab
"""
import os
import re
import shutil
import subprocess

import click

from numdifftools.info import __doc__ as INFO_TXT


ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'numdifftools'


def remove_previous_build():
    egginfo_path = os.path.join('src', '{}.egg-info'.format(PACKAGE_NAME))
    docs_folder = os.path.join(ROOT, 'docs', '_build')

    for dirname in ['dist', 'build', egginfo_path, docs_folder]:
        path = os.path.join(ROOT, dirname)
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("version")
def build_main(version):
    """Build and update {} version, documentation and package.

    The script remove the previous built binaries and generated documentation
    before it generate the documentation and build the binaries and finally
    check the built binaries.
    """.format(PACKAGE_NAME)
    remove_previous_build()
    set_package(version)
    update_readme()

    for cmd in ['docs', 'sdist', 'bdist_wheel', 'egg_info']:
        try:
            subprocess.call(["python", "setup.py", cmd])
        except Exception as error:  # subprocess.CalledProcessError:
            print('{}: {}'.format(cmd, str(error)))
    try:
        subprocess.call(["twine", "check", "dist/*"])
    except Exception as error:  # subprocess.CalledProcessError:
        print("Twine: ", str(error))


def update_readme():
    readme_txt = INFO_TXT.replace(
        """Introduction to {}
================{}
""".format(PACKAGE_NAME, '='*len(PACKAGE_NAME)), """{1}
{0}
{1}
""".format(PACKAGE_NAME, '='*len(PACKAGE_NAME)))

    filename = os.path.join(ROOT, "README.rst")
    with open(filename, "w") as fid:
        fid.write(readme_txt)


def set_package(version):
    """Set version of {} package""".format(PACKAGE_NAME)

    if version:
        filename = os.path.join(ROOT, "src", PACKAGE_NAME, "__init__.py")
        print("Version: {}".format(version))
        with open(filename, "r") as fid:
            text = fid.read()

        new_text = re.sub(r"__version__ = ['\"]([^'\"]*)['\"]",
                          '__version__ = "{}"'.format(version),
                          text, re.M)  # @UndefinedVariable

        with open(filename, "w") as fid:
            fid.write(new_text)


if __name__ == "__main__":
    build_main()
