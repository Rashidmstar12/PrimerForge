import os

test_dir = "tests"
for filename in os.listdir(test_dir):
    if filename.startswith("test_") and filename.endswith(".py"):
        filepath = os.path.join(test_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        needs_lgb = False
        needs_mappy = False
        
        if "ml_scorer" in content or "active_learning" in content or "continual_learner" in content or "optimizer" in content or "validation" in content:
            needs_lgb = True
        if "specificity" in content:
            needs_mappy = True
            
        guard = ""
        if needs_lgb:
            guard += """import pytest
try:
    import lightgbm
except ImportError:
    pytest.skip("lightgbm not available", allow_module_level=True)

"""
        if needs_mappy:
            guard += """import pytest
try:
    import mappy
except ImportError:
    pytest.skip("mappy not available", allow_module_level=True)

"""
        
        if guard:
            if "lightgbm not available" in content or "mappy not available" in content:
                print(f"Skipping {filename} - guard already exists")
                continue
                
            new_content = guard + content
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Added guard to {filename} (lgb={needs_lgb}, mappy={needs_mappy})")
