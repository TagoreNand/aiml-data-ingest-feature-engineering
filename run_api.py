import sys
sys.path.insert(0, '.')
import uvicorn
uvicorn.run('src.serving.api:app', host='0.0.0.0', port=8000, reload=False)
