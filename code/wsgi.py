# WSGI 入口, 供 gunicorn 加载: `gunicorn wsgi:app`
from predictor import app

if __name__ == "__main__":
    app.run()
