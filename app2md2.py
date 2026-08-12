import os
from datetime import datetime

MAX_FILE_SIZE = 1024 * 1024  # 1 MB


def process_folder(folder_path, max_file_size=MAX_FILE_SIZE):
    # --- No changes in this section ---
    if not os.path.isdir(folder_path):
        return "Invalid directory. Please enter a valid folder path."

    folder_name = os.path.basename(folder_path)
    md_content = f"# Code Files in Folder: {folder_name}\n\n"

    # List of file extensions to include
    valid_extensions = (
        ".py",
        ".js",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".rb",
        ".go",
        ".php",
        ".cs",
        ".ts",
        ".html",
        ".css",
        ".scss",
        ".xml",
        ".kt",
        ".swift",
        ".m",
        ".mm",
        ".sh",
        ".bat",
        ".ps1",
        ".r",
        ".pl",
        ".pm",
        ".asp",
        ".jsp",
        ".sql",
        ".lua",
        ".dart",
        ".erl",
        ".ex",
        ".jsx",
        ".tsx",
        ".scala",
        ".rs",
        ".vb",
        ".f90",
        ".for",
        ".s",
        ".ini",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".md",
        ".rst",
    )

    filenames_to_ignore = {
        "package-lock.json",
    }
    # List of specific filenames to always include
    specific_filenames_to_include = {
        "BUILD",
        ".bazelrc",
        ".gitignore",
        "makefile",
        "Makefile",
        "WORKSPACE",
        "gpp_build.sh",
        "Dockerfile",
    }

    # --- NEW: List of directories to ignore ---
    # Using a set for efficient lookup
    dirs_to_ignore = {
        ".git",
        "__pycache__",
        "node_modules",
        "venv",
        ".venv",
        "env",
        ".vscode",
        ".idea",
        "build",
        "dist",
        "target",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        ".next",
        ".cache",
        ".eggs",
        ".gitlab",
        ".svn",
        "coverage",
        "htmlcov",
        ".nox",
        ".terraform",
        ".serverless",
        "site",
        "_build",
    }

    # --- CHANGED: Use os.walk() for recursive traversal ---
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # --- NEW: Prune the directories to explore ---
        # This is an efficient way to prevent os.walk from descending into ignored folders.
        dirnames[:] = sorted([d for d in dirnames if d not in dirs_to_ignore])

        # --- NEW: Add a header for the current directory ---
        # We calculate the relative path to keep the output clean.
        relative_dir = os.path.relpath(dirpath, folder_path)

        # Don't create a header for the root directory if it's empty of relevant files,
        # so we'll collect content for this directory first.
        dir_md_content = ""

        # --- CHANGED: Loop through filenames from os.walk() ---
        for filename in sorted(filenames):
            if filename in filenames_to_ignore:
                continue
            # Check if the file should be included
            should_include = filename.endswith(valid_extensions) or filename in specific_filenames_to_include

            # Check if it's the output file we might be generating
            # Ignore previously generated summary files.
            is_output_file = filename == "code_summary.md" or filename.startswith("code_summary_")

            if should_include and not is_output_file:
                # --- CHANGED: Construct the full file path using dirpath ---
                file_path = os.path.join(dirpath, filename)
                if os.path.getsize(file_path) > max_file_size:
                    dir_md_content += (
                        f"### {os.path.relpath(file_path, folder_path)}\n\n"
                        f"*Skipped: file exceeds {max_file_size // 1024} KB limit*\n\n"
                    )
                    continue

                relative_file = os.path.relpath(file_path, folder_path)
                dir_md_content += f"### {relative_file}\n\n"
                # --- CHANGED: Use H3 for filename for better structure ---
                # dir_md_content += f"### {filename}\n\n"
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as code_file:
                        code = code_file.read()

                    _, ext = os.path.splitext(filename)
                    lang_hint = ext[1:] if ext else filename.lower()
                    if filename == ".gitignore":
                        lang_hint = "gitignore"
                    elif filename.lower() == "makefile":
                        lang_hint = "makefile"
                    elif filename == ".bazelrc":
                        lang_hint = "bazelrc"
                    elif filename in {"BUILD", "WORKSPACE"}:
                        lang_hint = "bazel"
                    elif filename == "Dockerfile":
                        lang_hint = "dockerfile"

                    dir_md_content += f"```{lang_hint}\n{code}\n```\n\n"
                except Exception as e:
                    dir_md_content += f"**Error reading file {filename}: {str(e)}**\n\n"

        # --- NEW: Add the directory content to the main markdown if it's not empty ---
        if dir_md_content:
            # Use '.' for the root directory for clarity
            display_path = relative_dir if relative_dir != "." else "root directory"
            md_content += f"## Directory: `{display_path}`\n\n"
            md_content += dir_md_content

    # --- No changes in this section ---
    output_filename = "code_summary.md"
    output_path = os.path.join(folder_path, output_filename)

    # Check if the base output file exists to avoid overwriting.
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"code_summary_{timestamp}.md"
        output_path = os.path.join(folder_path, output_filename)

    try:
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write(md_content)
        return f"Markdown file created at: {output_path}"
    except Exception as e:
        return f"Error writing markdown file: {str(e)}"


if __name__ == "__main__":
    folder_path = input("Enter the path of the folder to process: ").strip().strip('"')
    result = process_folder(folder_path)
    print(result)
