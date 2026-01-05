"""
Script to build the package.

The script removes the previous built binaries and generated documentation
before it generate the documentation and build the binaries and finally
check the built binaries.

It assumes that the library is installed in so called develop mode.

Created on 7. des. 2018

@author: pab
"""
import os
import re
import shutil
import subprocess
import importlib


ROOT = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = 'numdifftools'
INFO = importlib.import_module(PACKAGE_NAME+'.info', './src')
LICENSE = importlib.import_module(PACKAGE_NAME+'.license', './src')


def remove_previous_build():
    """Removes ./dist, ./build, ./docs/_build, and ./src/{}.egg-info directories.
    """.format(PACKAGE_NAME)
    egginfo_path = os.path.join('src', '{}.egg-info'.format(PACKAGE_NAME))
    docs_folder = os.path.join('docs', '_build')

    for dirname in ['dist', 'build', egginfo_path, docs_folder]:
        path = os.path.join(ROOT, dirname)
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


def update_readme():
    readme_txt = INFO.__doc__.replace(
        """Introduction to {}
================{}
""".format(PACKAGE_NAME, '='*len(PACKAGE_NAME)), """{1}
{0}
{1}
""".format(PACKAGE_NAME, '='*len(PACKAGE_NAME)))

    readme_txt = readme_txt.replace('.. only:: html', '')
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


def update_license():
    filename = os.path.join(ROOT, "LICENSE.txt")
    with open(filename, "w") as fid:
        fid.write(LICENSE.__doc__)


def call_subprocess(cmd_opts):
    """Safe call to subprocess"""
    print("\n\n***********************************************")
    print("Running {}".format(' '.join(cmd_opts)))
    try:
        subprocess.call(cmd_opts)
    except Exception as error:  # subprocess.CalledProcessError:
        print(str(error))
    print("***********************************************\n")


if __name__ == "__main__":
    import click

    @click.group(context_settings=dict(help_option_names=['-h', '--help']))
    def cli():
        """Main entry point for build and maintenance scripts."""
        pass

    @cli.command()
    @click.argument("version")
    def build(version):
        """Build and update {} version, documentation and package.

        The script remove the previous built binaries and generated documentation
        before it generate the documentation and build the binaries
        and finally check the built binaries.
        """.format(PACKAGE_NAME)
        remove_previous_build()
        
        set_package(version)
        update_license()
        update_readme()
        call_subprocess(["pdm", "run", ""])
        for cmd in ['docs-doctest', 'docs-html', 'docs-latex', 'docs-pdf', 'build']:
            call_subprocess(["pdm", "run", cmd])

    @cli.command("update-readme") # Define the new command
    def update_readme_cli():
        """Update the README.rst file from the INFO module docstring."""
        update_readme()

    @cli.command("update-license") # Define the new command
    def update_license_cli():
        """Update the LICENSE.txt file from the LICENSE module docstring."""
        update_license()

    cli()
