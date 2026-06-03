import sys
import os

# Add the project directory to sys.path
sys.path.insert(0, os.getcwd())

try:
    from app import app
    
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    
    missing = []
    for route in ["/diabetes", "/heart", "/obesity"]:
        if route not in rules:
            missing.append(route)
    
    if missing:
        print(f"FAILED: Missing routes: {missing}")
        sys.exit(1)
    else:
        print("SUCCESS: All routes found.")
        sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
