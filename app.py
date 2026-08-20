from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AWS Learning Backend"}

@app.get("/health")
def health():
    return {"status": "ok"}