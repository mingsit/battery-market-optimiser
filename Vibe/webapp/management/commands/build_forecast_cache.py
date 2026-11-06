"""Build the rolling ARIMA forecast cache used by the dashboard."""

from django.core.management.base import BaseCommand

from webapp import services


class Command(BaseCommand):
    help = (
        "Compute the rolling ARIMA forecast once and store it in "
        "webapp/cache/forecasts.pkl so page loads are fast."
    )

    def handle(self, *args, **options):
        self.stdout.write("Building the ARIMA forecast cache ...")
        result = services.build_and_cache_forecasts()
        self.stdout.write(
            self.style.SUCCESS(
                f"Cached {len(result.paths)} forecast origins "
                f"({result.horizon} half-hour steps each) at "
                f"{services.FORECAST_CACHE_PATH}."
            )
        )
