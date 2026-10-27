"""Context processors for AQI-GRU demonstration web application."""

from src.demo import get_aqi_reference_rows


def aqi_reference(request):
    """Expose official CPCB AQI reference categories to all templates."""
    return {
        "aqi_reference_rows": get_aqi_reference_rows(),
    }
