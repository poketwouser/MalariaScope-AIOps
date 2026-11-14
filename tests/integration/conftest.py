import os
import tempfile

# app.main opens the feedback store on import, keep it away from the real one
os.environ.setdefault("FEEDBACK_DIR", tempfile.mkdtemp())
