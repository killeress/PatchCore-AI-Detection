from pathlib import Path


def test_inference_trend_moves_high_percentage_labels_below_chart_legend():
    template = Path("templates/ric_report.html").read_text(encoding="utf-8")

    assert "Number.isFinite(value) && value >= 90 ? 'bottom' : 'top'" in template
    assert "clamp: true" in template


def test_report_comparison_trend_moves_high_percentage_labels_below_chart_legend():
    template = Path("templates/ric_report.html").read_text(encoding="utf-8")
    trend = template.split("function renderComparisonTrend()", 1)[1].split(
        "function computeReviewSummary()", 1
    )[0]

    assert "value >= 90) return 'bottom';" in trend
    assert "value <= 10) return 'top';" in trend
    assert "return context.datasetIndex === 2 ? 'bottom' : 'top';" in trend
    assert "clamp: true" in trend
