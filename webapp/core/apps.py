import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        """Load the model and scalers once at startup on CPU."""
        try:
            from src.demo import get_model_bundle
            get_model_bundle()
            logger.info("CoreConfig: AQI-GRU model bundle loaded successfully on CPU at startup.")
        except Exception as e:
            logger.warning("CoreConfig: Model bundle preload note: %s", e)
