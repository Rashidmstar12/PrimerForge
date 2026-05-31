import os

files_to_update = [
    "BIORXIV_MANUSCRIPT.md",
    "HOW_TO_USE_GUIDE.md",
    "README.md",
    "publication_package/BIORXIV_MANUSCRIPT.tex",
    "publication_package/documentation/README.md"
]

for file_path in files_to_update:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Replace pip install primerforge with pip install primerforge-py
        updated_content = content.replace("pip install primerforge", "pip install primerforge-py")
        
        # Also replace "primerforge" package installation references
        updated_content = updated_content.replace("register the tool name primerforge", "register the tool name primerforge-py")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated: {file_path}")
    else:
        print(f"Not found: {file_path}")
