import os
import sys

from flask import Flask
from flask_caching import Cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)  # parent dir, e.g. MasterScanner/ if this lives in MasterScanner/flask_ui/

sys.path.insert(0, _HERE)

# Prefer the real, live `utils/` package from the parent MasterScanner repo
# (when flask_ui/ is dropped in as a sibling of it). Fall back to the
# bundled legacy_utils/ snapshot copy if utils/ isn't found alongside —
# e.g. when running flask_ui/ standalone, unzipped on its own.
if os.path.isdir(os.path.join(_REPO_ROOT, "utils")):
    sys.path.insert(0, _REPO_ROOT)
else:
    _legacy_path = os.path.join(_HERE, "legacy_utils")
    if os.path.isdir(_legacy_path):
        if "utils" not in sys.modules:
            try:
                import legacy_utils  # noqa: F401
                sys.modules["utils"] = sys.modules["legacy_utils"]
            except Exception:
                pass

cache = Cache()


def create_app(config_object: str = "config.Config") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)
    cache.init_app(app)

    from routes.dashboard import bp as dashboard_bp
    from routes.stock_detail import bp as stock_detail_bp
    from routes.heatmap import bp as heatmap_bp
    from routes.lifecycle import bp as lifecycle_bp
    from routes.mobile import bp as mobile_bp
    from routes.stream import bp as stream_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(stock_detail_bp)
    app.register_blueprint(heatmap_bp)
    app.register_blueprint(lifecycle_bp)
    app.register_blueprint(mobile_bp)
    app.register_blueprint(stream_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
