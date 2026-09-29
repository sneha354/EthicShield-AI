"""
Root launcher for AI Ethics Auditor (EthicShield AI)
Starts FastAPI web server with Uvicorn on http://localhost:8000
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    print("==================================================================")
    print("       Starting EthicShield AI - AI Ethics & Safety Auditor       ")
    print("==================================================================")
    print("  Dashboard UI : http://localhost:8000")
    print("  Swagger Docs : http://localhost:8000/docs")
    print("==================================================================")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)

